import React, { cloneElement } from 'react';
import { useInfoView } from './InfoView';

/**
 * App-wide Tooltip.
 *
 * Help text is shown exclusively through the Info View (see
 * components/InfoView.js) — there are no popup tooltips on the controls:
 *
 *  • Info View ON  — on hover/focus the `title` is reported to the bottom
 *    Info View pill, so help shows in one fixed place rather than over the
 *    control itself.
 *  • Info View OFF — no hover help at all; the child renders untouched.
 *
 * The API matches MUI's Tooltip (drop-in for the existing call sites): pass a
 * `title` plus a single child element. Placement/arrow/delay props are
 * accepted but ignored, since there's no popup.
 */
export default function Tooltip({ children, title }) {
    const { enabled, setHint } = useInfoView();
    const child = React.Children.only(children);

    if (!enabled) {
        // No popup tooltips — the Info View is the only help surface.
        return child;
    }

    // Info View mode: route the tip to the bottom pill on hover/focus.
    return cloneElement(child, {
        onMouseEnter: (e) => { setHint(title); child.props?.onMouseEnter?.(e); },
        onMouseLeave: (e) => { setHint(null); child.props?.onMouseLeave?.(e); },
        onFocus: (e) => { setHint(title); child.props?.onFocus?.(e); },
        onBlur: (e) => { setHint(null); child.props?.onBlur?.(e); },
    });
}
