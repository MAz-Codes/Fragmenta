import React, { useState } from 'react';
import { lossChartStyles } from '../theme';

// Exponential moving average. alpha controls smoothness:
//   alpha → 1   = no smoothing (output equals input)
//   alpha → 0   = heavy smoothing (output flat-ish line)
// Diffusion loss is intrinsically noisy because each step samples a random
// timestep with different difficulty, so a small alpha (heavy smoothing) is
// what makes the underlying trend visible.
const EMA_ALPHA = 0.06;

function smoothEMA(values, alpha = EMA_ALPHA) {
    if (values.length === 0) return [];
    const out = [values[0]];
    for (let i = 1; i < values.length; i++) {
        out.push(alpha * values[i] + (1 - alpha) * out[i - 1]);
    }
    return out;
}

export default function LossChart({ data, width = 600, height = 200 }) {
    const [hover, setHover] = useState(null);
    const padding = lossChartStyles.padding;
    const colors = lossChartStyles.colors;
    const axisFontSize = lossChartStyles.axisFontSize;
    const tooltip = lossChartStyles.tooltip;

    if (!data || data.length === 0) return null;

    const innerW = width - padding.left - padding.right;
    const innerH = height - padding.top - padding.bottom;

    const xs = data.map(d => d.step);
    const ys = data.map(d => d.loss);
    const smoothed = smoothEMA(ys);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;

    const xScale = v => padding.left + ((v - xMin) / xRange) * innerW;
    const yScale = v => padding.top + innerH - ((v - yMin) / yRange) * innerH;

    const points = data.map(d => `${xScale(d.step)},${yScale(d.loss)}`).join(' ');
    const smoothedPoints = data.map((d, i) => `${xScale(d.step)},${yScale(smoothed[i])}`).join(' ');

    const yTicks = 4;
    const xTicks = Math.min(5, data.length);

    const handleMove = (e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const px = ((e.clientX - rect.left) / rect.width) * width;
        let nearest = data[0];
        let bestDist = Infinity;
        for (const d of data) {
            const dist = Math.abs(xScale(d.step) - px);
            if (dist < bestDist) { bestDist = dist; nearest = d; }
        }
        setHover(nearest);
    };

    return (
        <svg
            viewBox={`0 0 ${width} ${height}`}
            preserveAspectRatio="none"
            style={lossChartStyles.svg}
            onMouseMove={handleMove}
            onMouseLeave={() => setHover(null)}
        >
            {Array.from({ length: yTicks + 1 }, (_, i) => {
                const v = yMin + (yRange * i) / yTicks;
                const y = yScale(v);
                return (
                    <g key={`y${i}`}>
                        <line x1={padding.left} x2={width - padding.right} y1={y} y2={y}
                            stroke={colors.grid} strokeDasharray="3 3" />
                        <text x={padding.left - 6} y={y + 4} textAnchor="end"
                            fontSize={axisFontSize} fill={colors.axis}>{v.toFixed(3)}</text>
                    </g>
                );
            })}

            {Array.from({ length: xTicks }, (_, i) => {
                const v = xMin + (xRange * i) / Math.max(xTicks - 1, 1);
                const x = xScale(v);
                return (
                    <text key={`x${i}`} x={x} y={height - padding.bottom + 16}
                          textAnchor="middle" fontSize={axisFontSize} fill={colors.axis}>
                        {Math.round(v)}
                    </text>
                );
            })}

            {/* Raw loss as a faint background trace (the "peaky" curve). */}
            <polyline fill="none" stroke={colors.line} strokeWidth="1" strokeOpacity="0.25" points={points} />

            {/* EMA-smoothed loss as the primary trace. */}
            <polyline fill="none" stroke={colors.line} strokeWidth="2.5" points={smoothedPoints} />

            {hover && (
                <g>
                    <line x1={xScale(hover.step)} x2={xScale(hover.step)}
                          y1={padding.top} y2={height - padding.bottom}
                          stroke={colors.axis} strokeDasharray="2 2" />
                    <circle cx={xScale(hover.step)} cy={yScale(hover.loss)} r="3" fill={colors.line} fillOpacity="0.4" />
                    <circle cx={xScale(hover.step)} cy={yScale(smoothed[data.indexOf(hover)] ?? hover.loss)} r="4" fill={colors.line} />
                    <g transform={`translate(${Math.min(xScale(hover.step) + 8, width - (tooltip.width + 10))}, ${padding.top + 6})`}>
                        <rect width={tooltip.width} height={tooltip.height + 12} rx={tooltip.rx} fill={colors.tooltipBg} stroke={colors.tooltipBorder} />
                        <text x={tooltip.textX} y={tooltip.timeY} fontSize={axisFontSize} fill={colors.tooltipText}>Step: {hover.step}</text>
                        <text x={tooltip.textX} y={tooltip.lossY} fontSize={axisFontSize} fill={colors.tooltipText}>Raw: {hover.loss.toFixed(4)}</text>
                        <text x={tooltip.textX} y={tooltip.lossY + 14} fontSize={axisFontSize} fill={colors.tooltipText}>Smoothed: {(smoothed[data.indexOf(hover)] ?? hover.loss).toFixed(4)}</text>
                    </g>
                </g>
            )}
        </svg>
    );
}
