import React, { useState } from 'react';
import { lossChartStyles } from '../theme';

function fmtTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
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

    const xs = data.map(d => d.time);
    const ys = data.map(d => d.loss);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;

    const xScale = v => padding.left + ((v - xMin) / xRange) * innerW;
    const yScale = v => padding.top + innerH - ((v - yMin) / yRange) * innerH;

    const points = data.map(d => `${xScale(d.time)},${yScale(d.loss)}`).join(' ');

    const yTicks = 4;
    const xTicks = Math.min(5, data.length);

    const handleMove = (e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const px = ((e.clientX - rect.left) / rect.width) * width;
        let nearest = data[0];
        let bestDist = Infinity;
        for (const d of data) {
            const dist = Math.abs(xScale(d.time) - px);
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
                        {fmtTime(v)}
                    </text>
                );
            })}

            <polyline fill="none" stroke={colors.line} strokeWidth="2" points={points} />

            {data.map((d, i) => (
                <circle key={i} cx={xScale(d.time)} cy={yScale(d.loss)} r="2" fill={colors.point} />
            ))}

            {hover && (
                <g>
                    <line x1={xScale(hover.time)} x2={xScale(hover.time)}
                          y1={padding.top} y2={height - padding.bottom}
                          stroke={colors.axis} strokeDasharray="2 2" />
                    <circle cx={xScale(hover.time)} cy={yScale(hover.loss)} r="4" fill={colors.line} />
                    <g transform={`translate(${Math.min(xScale(hover.time) + 8, width - (tooltip.width + 10))}, ${padding.top + 6})`}>
                        <rect width={tooltip.width} height={tooltip.height} rx={tooltip.rx} fill={colors.tooltipBg} stroke={colors.tooltipBorder} />
                        <text x={tooltip.textX} y={tooltip.timeY} fontSize={axisFontSize} fill={colors.tooltipText}>Time: {fmtTime(hover.time)}</text>
                        <text x={tooltip.textX} y={tooltip.lossY} fontSize={axisFontSize} fill={colors.tooltipText}>Loss: {hover.loss.toFixed(4)}</text>
                    </g>
                </g>
            )}
        </svg>
    );
}
