import React, { useState, useMemo, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';

// --- Type Definitions ---

interface DataPoint {
  time: number; // microseconds
  voltage: number; // Volts
}

interface SignalMetrics {
  frequencyKhz: number;
  beta: number;
  lambda: number;
  periodUs: number;
  validCycles: number;
}

interface FitParams {
  A: number;
  beta: number;
  C: number;
}

// --- Utils: Signal Processing ---

const parseFileContent = (content: string): DataPoint[] => {
  const lines = content.split('\n');
  const data: DataPoint[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    // Skip lines that look like metadata headers (usually start with quotes)
    if (trimmed.startsWith('"') || trimmed.startsWith("'")) continue;

    // Replace comma with dot for proper float parsing
    const cleanLine = trimmed.replace(/,/g, '.');
    // Split by whitespace (handles tabs and spaces)
    const parts = cleanLine.split(/\s+/);
    
    // We need at least Time and one Voltage
    if (parts.length >= 2) {
      const t = parseFloat(parts[0]);
      const v1 = parseFloat(parts[1]);
      
      // Basic validity check
      if (!Number.isFinite(t) || !Number.isFinite(v1)) continue;

      // Filter out scope clipping/error values (often +/- 1E+308)
      if (Math.abs(v1) > 1e30) continue;

      let voltage = v1;

      // Handle Envelope mode (3 columns: Time, Min, Max)
      // If we have a 3rd column that is a valid number, we average V1 (Min) and V2 (Max)
      // to estimate the signal center.
      if (parts.length >= 3) {
         const v2 = parseFloat(parts[2]);
         if (Number.isFinite(v2) && Math.abs(v2) < 1e30) {
             voltage = (v1 + v2) / 2;
         }
      }

      data.push({ time: t, voltage: voltage });
    }
  }
  return data;
};

// Simple Moving Average Filter
const smoothSignal = (values: number[], windowSize: number): number[] => {
  if (windowSize <= 1) return values;
  const smoothed: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - Math.floor(windowSize / 2));
    const end = Math.min(values.length, i + Math.floor(windowSize / 2) + 1);
    let sum = 0;
    for (let j = start; j < end; j++) {
      sum += values[j];
    }
    smoothed.push(sum / (end - start));
  }
  return smoothed;
};

const processSignal = (rawData: DataPoint[], smoothingWindow: number, periodTolerancePct: number, invert: boolean) => {
  if (rawData.length === 0) return null;

  // 1. Prepare data (Units: µs, V)
  const t_raw = rawData.map(d => d.time * 1e6);
  let u_original = rawData.map(d => d.voltage);

  // --- MANUAL INVERSION LOGIC ---
  if (invert) {
      u_original = u_original.map(v => -v);
  }

  // 2. Apply Smoothing (Noise Filter)
  const u_processed = smoothSignal(u_original, smoothingWindow);

  // 3. Determine DC Offset (Mean)
  // FIX: Average the last 20% of the signal.
  // This assumes the signal settles to zero (or DC bias) at the end.
  // This is critical for high-damping signals where the start is a huge impulse.
  const tailSampleCount = Math.max(Math.floor(u_processed.length * 0.20), 10);
  const tailData = u_processed.slice(-tailSampleCount);
  const meanVoltage = tailData.reduce((a, b) => a + b, 0) / tailData.length;

  // 4. Find ALL Potential Rising Zero Crossings
  const rawCrossings: { t: number, idx: number }[] = [];
  
  for (let i = 0; i < u_processed.length - 1; i++) {
    if (u_processed[i] <= meanVoltage && u_processed[i+1] > meanVoltage) {
      const y1 = u_processed[i];
      const y2 = u_processed[i+1];
      const t1 = t_raw[i];
      const t2 = t_raw[i+1];
      const fraction = (meanVoltage - y1) / (y2 - y1);
      const t_cross = t1 + fraction * (t2 - t1);
      
      rawCrossings.push({ t: t_cross, idx: i });
    }
  }

  // 5. Apply Period Locking Logic
  const validCrossings: { t: number, idx: number }[] = [];
  
  if (rawCrossings.length >= 2) {
      validCrossings.push(rawCrossings[0]);
      validCrossings.push(rawCrossings[1]);

      const T_ref = rawCrossings[1].t - rawCrossings[0].t;

      for (let i = 2; i < rawCrossings.length; i++) {
          const lastValid = validCrossings[validCrossings.length - 1];
          const current = rawCrossings[i];
          const currentPeriod = current.t - lastValid.t;
          const deviation = Math.abs(currentPeriod - T_ref);
          const allowedDeviation = T_ref * periodTolerancePct;

          if (deviation <= allowedDeviation) {
              validCrossings.push(current);
          } else {
              break; 
          }
      }
  }

  // 6. Detect peaks only within VALID intervals
  const peaks: {x: number, y: number}[] = [];
  
  if (validCrossings.length >= 2) {
    for (let i = 0; i < validCrossings.length - 1; i++) {
      const startIdx = validCrossings[i].idx;
      const endIdx = validCrossings[i+1].idx;
      
      let maxVal = -Infinity;
      let maxIdx = -1;

      for (let j = startIdx; j <= endIdx; j++) {
        if (u_processed[j] > maxVal) {
          maxVal = u_processed[j];
          maxIdx = j;
        }
      }

      if (maxIdx !== -1 && maxVal > meanVoltage) {
         peaks.push({ x: t_raw[maxIdx], y: maxVal });
      }
    }
  }

  // 7. Calculate Average Period
  let avgPeriodUs = 0;
  if (validCrossings.length >= 2) {
      let sumDiff = 0;
      for (let i = 0; i < validCrossings.length - 1; i++) {
          sumDiff += (validCrossings[i+1].t - validCrossings[i].t);
      }
      avgPeriodUs = sumDiff / (validCrossings.length - 1);
  }

  // 8. NORMALIZATION (Shift to 0,0)
  // We align the first valid zero crossing to t=0, and the mean voltage to u=0.
  let t_offset = 0;
  let u_offset = meanVoltage;

  if (validCrossings.length > 0) {
      t_offset = validCrossings[0].t;
  } else if (t_raw.length > 0) {
      t_offset = t_raw[0]; // Fallback
  }

  const t_norm = t_raw.map(t => t - t_offset);
  const u_raw_norm = u_original.map(u => u - u_offset);
  const u_smooth_norm = u_processed.map(u => u - u_offset);
  const peaks_norm = peaks.map(p => ({ x: p.x - t_offset, y: p.y - u_offset }));
  const crossings_norm = validCrossings.map(c => ({ t: c.t - t_offset, idx: c.idx }));

  return {
    t: t_norm,
    u_raw: u_raw_norm,
    u_smooth: u_smooth_norm,
    peaks: peaks_norm,
    zeroCrossings: crossings_norm, 
    meanVoltage: 0, // Normalized mean is now 0 (relative)
    avgPeriodUs,
    isInverted: invert
  };
};

const fitExponentialCurve = (peaks: {x: number, y: number}[], meanVoltage: number): FitParams | null => {
  // Allow fit with at least 2 peaks instead of 3.
  if (peaks.length < 2) return null;

  const C = meanVoltage; 

  let sumT = 0;
  let sumLnY = 0;
  let sumTLnY = 0;
  let sumT2 = 0;
  let count = 0;

  for (const p of peaks) {
    if (p.y > C) {
      const lnY = Math.log(p.y - C);
      sumT += p.x;
      sumLnY += lnY;
      sumTLnY += p.x * lnY;
      sumT2 += p.x * p.x;
      count++;
    }
  }

  if (count < 2) return null;

  const denominator = (count * sumT2 - sumT * sumT);
  if (denominator === 0) return null;

  const beta = (count * sumTLnY - sumT * sumLnY) / denominator;
  const lnA = (sumLnY - beta * sumT) / count;
  const A = Math.exp(lnA);

  return { A, beta, C };
};

// --- Components ---

const App = () => {
  const [data, setData] = useState<DataPoint[] | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  
  // Settings State
  const [smoothingWindow, setSmoothingWindow] = useState<number>(10);
  const [periodTolerance, setPeriodTolerance] = useState<number>(10); // Percent
  const [isInverted, setIsInverted] = useState<boolean>(false);

  // --- Processing Hook ---
  const processedData = useMemo(() => {
    if (!data) return null;
    return processSignal(data, smoothingWindow, periodTolerance / 100, isInverted);
  }, [data, smoothingWindow, periodTolerance, isInverted]);

  const fitResults = useMemo(() => {
    if (!processedData || processedData.peaks.length < 2) return null;
    
    const { peaks, meanVoltage, avgPeriodUs } = processedData;

    const fit = fitExponentialCurve(peaks, meanVoltage);
    
    let metrics: SignalMetrics | null = null;
    if (fit && avgPeriodUs > 0) {
      const f_hz = 1 / (avgPeriodUs * 1e-6);
      const lambda = Math.abs(fit.beta) * avgPeriodUs;
      // Beta calculation: Lambda / Period (in seconds)
      const betaSeconds = lambda / (avgPeriodUs * 1e-6);

      metrics = {
        frequencyKhz: f_hz / 1000,
        beta: betaSeconds,
        lambda: lambda,
        periodUs: avgPeriodUs,
        validCycles: peaks.length
      };
    }

    return { fit, metrics };
  }, [processedData]);

  // --- File Handler ---
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setError(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const parsed = parseFileContent(text);
        if (parsed.length === 0) {
          setError("Nie znaleziono poprawnych danych numerycznych w pliku.");
          setData(null);
        } else {
          setData(parsed);
          setIsInverted(false); 
        }
      } catch (err) {
        setError("Błąd podczas odczytu pliku.");
      }
    };
    reader.readAsText(file);
  };

  // --- Plotly Refs ---
  const plotRef = useRef<HTMLDivElement>(null);
  const hiddenReportRef = useRef<HTMLDivElement>(null); // Ref for hidden report generation

  // --- REPORT GENERATION ---
  const downloadReport = async () => {
      if (!processedData || !hiddenReportRef.current) return;

      const { t, u_smooth, peaks } = processedData;
      const fit = fitResults?.fit;

      // Trace 1: Smoothed line, RED.
      const traceReport = {
          x: t,
          y: u_smooth,
          mode: 'lines',
          line: { color: 'red', width: 2 },
          showlegend: false
      };

      const traces = [traceReport];

      // Calculate Range for Y-Axis manually to ensure fit
      const allY = [...u_smooth];

      // Trace 2: Approximation
      if (fit && peaks.length > 0) {
        const startT = peaks[0].x;
        const endT = t[t.length-1]; 
        
        const step = (endT - startT) / 200;
        const fitX = [];
        const fitY = [];
        for(let val=startT; val<=endT; val+=step) {
            fitX.push(val);
            const yVal = fit.A * Math.exp(fit.beta * val) + fit.C;
            fitY.push(yVal);
            allY.push(yVal); // Include fit values in range calculation
        }

        const traceFit = {
            x: fitX,
            y: fitY,
            mode: 'lines',
            line: { color: 'black', width: 2, dash: 'dash' },
            showlegend: false
        };
        traces.push(traceFit);
      }

      // Explicitly calculate min/max to force Plotly to adapt to inverted data
      let yMin = Math.min(...allY);
      let yMax = Math.max(...allY);
      let padding = (yMax - yMin) * 0.1;
      if (padding === 0) padding = 1;

      const layoutReport = {
          title: '', 
          plot_bgcolor: 'white',
          paper_bgcolor: 'white',
          xaxis: {
              title: 'Czas [µs]',
              titlefont: { color: 'black' },
              tickfont: { color: 'black' },
              gridcolor: '#e0e0e0',
              zerolinecolor: 'black',
              showgrid: true,
              showticklabels: true, 
              ticks: 'outside',
              tickwidth: 1,
              ticklen: 5,
              showline: true,
              linewidth: 1,
              linecolor: 'black',
              range: [0, Math.max(...t)],
              automargin: true
          },
          yaxis: {
              title: 'Napięcie [V]',
              titlefont: { color: 'black' },
              tickfont: { color: 'black' },
              gridcolor: '#e0e0e0',
              zerolinecolor: 'black',
              showgrid: true,
              showticklabels: true,
              ticks: 'outside',
              tickwidth: 1,
              ticklen: 5,
              showline: true,
              linewidth: 1,
              linecolor: 'black',
              range: [yMin - padding, yMax + padding], // Force correct range
              automargin: true
          },
          showlegend: false, 
          margin: { l: 70, r: 20, t: 20, b: 70 }, 
          width: 800, // Reduced from 1000 for better Word doc fit
          height: 500 // Reduced from 600
      };

      // @ts-ignore
      await window.Plotly.newPlot(hiddenReportRef.current, traces, layoutReport);
      
      // @ts-ignore
      await window.Plotly.downloadImage(hiddenReportRef.current, {
          format: 'png',
          filename: 'sygnal_raport_inzynierski',
          height: 500,
          width: 800,
          scale: 2 // Scale 2x ensures high quality (Retina) while logical size is small
      });
  };


  // --- Main Live Chart Effect ---
  useEffect(() => {
    if (!processedData || !plotRef.current) return;

    const { t, u_raw, u_smooth, peaks, zeroCrossings, meanVoltage } = processedData;
    const fit = fitResults?.fit;

    // Trace 1: Original Signal
    const traceRaw = {
      x: t,
      y: u_raw,
      mode: 'lines',
      name: 'Sygnał',
      line: { color: '#004444', width: 1 },
      opacity: 0.5,
      hoverinfo: 'skip'
    };

    // Trace 2: Smoothed Signal
    const traceSmooth = {
      x: t,
      y: u_smooth,
      mode: 'lines',
      name: 'Wygładzony',
      line: { color: '#00f3ff', width: 2 }
    };

    // Trace 3: Valid Zero Crossings
    const traceZero = {
        x: zeroCrossings.map(z => z.t),
        y: zeroCrossings.map(() => meanVoltage),
        mode: 'markers',
        name: 'Punkt "0,0" (Start)',
        marker: { color: '#ffffff', size: 8, symbol: 'cross', line: {width: 2} }
    };

    // Trace 4: Detected Peaks (Filtered)
    const tracePeaks = {
      x: peaks.map(p => p.x),
      y: peaks.map(p => p.y),
      mode: 'markers',
      name: 'Szczyty',
      marker: { color: '#b87333', size: 12, symbol: 'diamond-open', line: { width: 3, color: '#ffaa00' } }
    };

    const traces: any[] = [traceRaw, traceSmooth, traceZero, tracePeaks];

    // Trace 5: Exponential Fit
    if (fit && peaks.length > 0) {
        const startT = peaks[0].x;
        const endT = t[t.length-1]; 
        
        const step = (endT - startT) / 200;
        const fitX = [];
        const fitY = [];
        for(let val=startT; val<=endT; val+=step) {
            fitX.push(val);
            fitY.push(fit.A * Math.exp(fit.beta * val) + fit.C);
        }

        const traceFit = {
            x: fitX,
            y: fitY,
            mode: 'lines',
            name: 'Aproksymacja',
            line: { color: '#ff0055', width: 3, dash: 'dot' }
        };
        traces.push(traceFit);
    }

    const shapeLine = {
        type: 'line',
        x0: t[0],
        y0: meanVoltage,
        x1: t[t.length-1],
        y1: meanVoltage,
        line: { color: '#b87333', width: 1, dash: 'dash' },
        opacity: 0.5
    };

    const cutoffTime = zeroCrossings.length > 0 ? zeroCrossings[zeroCrossings.length - 1].t : t[0];
    const shapeCutoff = {
        type: 'line',
        x0: cutoffTime,
        y0: Math.min(...u_smooth),
        x1: cutoffTime,
        y1: Math.max(...u_smooth),
        line: { color: '#ff0055', width: 2, dash: 'dot' },
        opacity: 0.7
    };


    const layout = {
      title: isInverted ? 'ANALIZA SYGNAŁU (ODWRÓCONY)' : 'ANALIZA SYGNAŁU (NORMALNY)',
      titlefont: { family: 'Share Tech Mono', size: 24, color: '#b87333' },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(10,10,10,0.8)',
      xaxis: {
        title: 'Czas (Znormalizowany) [µs]',
        titlefont: { color: '#b87333' },
        tickfont: { color: '#00f3ff' },
        gridcolor: '#333333',
        zerolinecolor: '#b87333'
      },
      yaxis: {
        title: 'Napięcie (Offset skorygowany) [V]',
        titlefont: { color: '#b87333' },
        tickfont: { color: '#00f3ff' },
        gridcolor: '#333333',
        zerolinecolor: '#b87333'
      },
      legend: {
        font: { color: '#e0e0e0' },
        bgcolor: 'rgba(0,0,0,0.5)',
        bordercolor: '#b87333',
        borderwidth: 1,
        orientation: 'h',
        y: -0.2
      },
      shapes: [shapeLine, shapeCutoff],
      margin: { l: 60, r: 40, t: 80, b: 80 },
      autosize: true
    };

    // @ts-ignore
    window.Plotly.newPlot(plotRef.current, traces, layout, {
        responsive: true,
        displayModeBar: true,
        toImageButtonOptions: {
            format: 'png',
            filename: 'signal_analysis_tool',
            height: 600,
            width: 1000,
            scale: 2
        }
    });

  }, [processedData, fitResults, isInverted]);


  return (
    <div className="flex flex-col md:flex-row min-h-screen">
      {/* Hidden Div for Report Generation */}
      <div ref={hiddenReportRef} style={{ display: 'none' }} />

      {/* Sidebar Panel */}
      <aside className="w-full md:w-80 bg-[#0b0c10] border-r-2 border-[#b87333] p-6 flex flex-col gap-6 relative">
        <div className="absolute top-0 right-0 p-2 opacity-30 pointer-events-none">
             <svg width="100" height="100" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="40" stroke="#b87333" strokeWidth="1" fill="none" strokeDasharray="5,5" />
                <path d="M50 10 L50 90 M10 50 L90 50" stroke="#b87333" strokeWidth="1" />
             </svg>
        </div>

        <div>
          <h1 className="text-3xl font-bold text-[#00f3ff] uppercase mb-2 border-b-2 border-[#b87333] pb-2 glow-text">
            Signal<br/>Analyzer<br/><span className="text-xl text-[#b87333]">2077</span>
          </h1>
          <p className="text-sm text-gray-400 mt-4">
            Wizualizacja i analiza danych
          </p>
        </div>

        <div className="mt-4">
            <label className="block text-[#b87333] text-sm font-bold uppercase mb-2 tracking-wider">
                Panel Kontrolny
            </label>
            <div className="relative border border-dashed border-[#b87333] p-4 rounded bg-[#0f0f0f] hover:bg-[#151515] transition-colors cursor-pointer group mb-6">
                <input 
                    type="file" 
                    accept=".txt,.csv" 
                    onChange={handleFileUpload}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                <div className="text-center">
                    <div className="text-2xl mb-2 group-hover:text-[#00f3ff] transition-colors">📂</div>
                    <span className="text-xs uppercase text-gray-300 group-hover:text-white">
                        {fileName || "Wybierz plik"}
                    </span>
                </div>
            </div>

            {/* Signal Processing Controls */}
            <div className="border-t border-[#333] pt-4 flex flex-col gap-4">
                
                {/* INVERT TOGGLE */}
                <div className="bg-[#111] p-3 border border-[#b87333] rounded flex items-center justify-between">
                    <span className="text-[#00f3ff] text-xs uppercase">Odwróć Sygnał</span>
                    <label className="relative inline-flex items-center cursor-pointer">
                        <input 
                            type="checkbox" 
                            className="sr-only peer" 
                            checked={isInverted} 
                            onChange={(e) => setIsInverted(e.target.checked)} 
                        />
                        <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[#b87333]"></div>
                    </label>
                </div>

                <div className="bg-[#111] p-3 border border-[#b87333] rounded">
                    <label className="flex justify-between text-[#00f3ff] text-xs uppercase mb-1">
                        <span>Filtr Wygładzający</span>
                        <span className="text-[#b87333]">{smoothingWindow} pkt</span>
                    </label>
                    <input 
                        type="range" 
                        min="1" 
                        max="50" 
                        step="1"
                        value={smoothingWindow} 
                        onChange={(e) => setSmoothingWindow(parseInt(e.target.value))}
                        className="w-full accent-[#b87333] cursor-pointer h-2 bg-[#333] rounded-lg appearance-none"
                    />
                </div>

                <div className="bg-[#111] p-3 border border-[#b87333] rounded">
                    <label className="flex justify-between text-[#00f3ff] text-xs uppercase mb-1">
                        <span>Tolerancja Okresu</span>
                        <span className="text-[#b87333]">±{periodTolerance}%</span>
                    </label>
                    <input 
                        type="range" 
                        min="5" 
                        max="50" 
                        step="1"
                        value={periodTolerance} 
                        onChange={(e) => setPeriodTolerance(parseInt(e.target.value))}
                        className="w-full accent-[#b87333] cursor-pointer h-2 bg-[#333] rounded-lg appearance-none"
                    />
                    <div className="text-[10px] text-gray-500 mt-1">
                        Przerywa analizę, jeśli kolejny okres różni się od T1 o więcej niż {periodTolerance}%.
                    </div>
                </div>

                {/* Download Report Button */}
                {data && (
                    <button 
                        onClick={downloadReport}
                        className="w-full mt-4 bg-[#222] border border-[#00f3ff] text-[#00f3ff] p-3 rounded uppercase font-bold text-sm hover:bg-[#00f3ff] hover:text-black transition-all shadow-[0_0_10px_rgba(0,243,255,0.2)]"
                    >
                        Pobierz Wykres (Raport)
                    </button>
                )}
            </div>

            {error && <div className="mt-2 text-[#ff0055] text-xs font-bold">{error}</div>}
        </div>

        <div className="mt-auto pt-6 border-t border-[#333]">
            <div className="text-[10px] text-[#555] font-mono">
                SYS.VERSION: 4.5.1 (REPORT-FIT)<br/>
                STATUS: ONLINE<br/>
                MEMORY: OPTIMAL
            </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-6 md:p-8 flex flex-col gap-8 relative overflow-hidden">
        {/* Background Grid Decoration */}
        <div className="absolute inset-0 pointer-events-none opacity-5" 
             style={{backgroundImage: 'linear-gradient(#00f3ff 1px, transparent 1px), linear-gradient(90deg, #00f3ff 1px, transparent 1px)', backgroundSize: '50px 50px'}}>
        </div>

        {!data ? (
             <div className="flex-1 flex flex-col items-center justify-center text-center opacity-50 border-2 border-dashed border-[#333] rounded-lg m-10">
                <div className="text-6xl mb-4 text-[#b87333] animate-pulse">⚡</div>
                <h2 className="text-2xl text-[#00f3ff] uppercase tracking-widest">System Standby</h2>
                <p className="mt-2 text-[#b87333]">Oczekiwanie na strumień danych...</p>
             </div>
        ) : (
            <>
                {/* Metrics Row */}
                <section>
                    <div className="flex items-center gap-3 mb-4 border-l-4 border-[#b87333] pl-3">
                         <h3 className="text-[#00f3ff] uppercase text-xl">Parametry Sygnału</h3>
                    </div>

                    {fitResults?.metrics ? (
                        <div className="bg-[#0b0c10] border border-[#b87333] p-5 rounded relative overflow-hidden">
                            <div className="absolute top-0 right-0 p-4 opacity-10">
                                <div className="text-6xl text-[#b87333]">λ</div>
                            </div>
                            <ul className="flex flex-col gap-2 text-lg">
                                <li className="flex items-baseline gap-2">
                                    <span className="text-[#b87333]">Częstotliwość (f):</span>
                                    <span className="text-[#00f3ff] font-bold">{fitResults.metrics.frequencyKhz.toFixed(2)} kHz</span>
                                </li>
                                <li className="flex items-baseline gap-2">
                                     <span className="text-[#b87333]">Logarytmiczny dekrement tłumienia (λ):</span>
                                     <span className="text-[#00f3ff] font-bold">{fitResults.metrics.lambda.toFixed(4)}</span>
                                </li>
                                 <li className="flex items-baseline gap-2">
                                     <span className="text-[#b87333]">Tłumienie (β):</span>
                                     <span className="text-[#00f3ff] font-bold">{fitResults.metrics.beta.toFixed(4)} s⁻¹</span>
                                </li>
                                <li className="flex items-baseline gap-2">
                                     <span className="text-[#b87333]">Okres (T):</span>
                                     <span className="text-[#00f3ff] font-bold">{fitResults.metrics.periodUs.toFixed(2)} µs</span>
                                </li>
                                <li className="flex items-baseline gap-2">
                                     <span className="text-[#b87333]">Poprawne Cykle:</span>
                                     <span className="text-[#00f3ff] font-bold">{fitResults.metrics.validCycles}</span>
                                </li>
                            </ul>
                        </div>
                    ) : (
                        <div className="text-[#ff0055] border border-[#ff0055] p-2 inline-block rounded text-sm bg-[#ff005520]">
                            ⚠ Sygnał niestabilny lub zbyt zaszumiony. Zwiększ tolerancję lub sprawdź inwersję.
                        </div>
                    )}
                </section>

                {/* Chart Section */}
                <section className="flex-1 flex flex-col min-h-[500px]">
                    <h3 className="text-[#00f3ff] uppercase text-xl mb-4 border-l-4 border-[#b87333] pl-3">Wizualizacja</h3>
                    <div className="flex-1 bg-[#0b0c10] border-2 border-[#b87333] rounded shadow-lg p-2 relative">
                        <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-[#00f3ff]"></div>
                        <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-[#00f3ff]"></div>
                        <div ref={plotRef} className="w-full h-full min-h-[450px]" />
                    </div>
                </section>
                
                {/* Raw Data Preview (Optional) */}
                <section>
                    <details className="cursor-pointer group">
                        <summary className="text-[#b87333] hover:text-[#00f3ff] transition-colors uppercase text-sm select-none">
                            [+] Pokaż podgląd surowych danych (Wczytano: {data.length} | Pierwsze 10)
                        </summary>
                        <div className="mt-4 overflow-x-auto bg-[#111] border border-[#333] p-4 rounded font-mono text-xs">
                            <table className="w-full text-left">
                                <thead>
                                    <tr className="border-b border-[#333] text-[#00f3ff]">
                                        <th className="p-2">Index</th>
                                        <th className="p-2">Time (Original)</th>
                                        <th className="p-2">Voltage [V]</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {data.slice(0, 10).map((row, i) => (
                                        <tr key={i} className="border-b border-[#222] text-gray-400 hover:bg-[#222]">
                                            <td className="p-2 text-[#b87333]">{i}</td>
                                            <td className="p-2">{row.time.toExponential(4)}</td>
                                            <td className="p-2">{row.voltage.toFixed(6)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </details>
                </section>
            </>
        )}
      </main>
    </div>
  );
};

const container = document.getElementById('root');
const root = createRoot(container!);
root.render(<App />);