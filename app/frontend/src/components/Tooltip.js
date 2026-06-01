import React, { cloneElement, useState } from 'react';
import { Tooltip as MuiTooltip } from '@mui/material';
import { useInfoView } from './InfoView';

/**
 * App-wide Tooltip.
 *
 * Two modes, switched by the Info View toggle (see components/InfoView.js):
 *
 *  • Info View OFF (default) — a thin wrapper around MUI's Tooltip that also
 *    dismisses the tooltip the instant its child is activated (clicked /
 *    key-pressed). Stock MUI keeps the tooltip open after a click because the
 *    control retains hover + focus, so help text lingers over a button you've
 *    already pressed. We take control of `open` and force it shut on click,
 *    then let the normal hover/focus listeners reopen it.
 *
 *  • Info View ON — no popup. On hover/focus the `title` is reported to the
 *    bottom Info View bar instead, so help text shows in one fixed place
 *    rather than over the control itself.
 *
 * The API is identical to MUI's Tooltip — every prop (title, placement,
 * arrow, enterDelay, …) passes straight through, and the child's own handlers
 * are preserved. Drop-in replacement for `import { Tooltip } from '@mui/material'`.
 */
export default function Tooltip({ children, title, ...props }) {
    const [open, setOpen] = useState(false);
    const { enabled, setHint } = useInfoView();
    const child = React.Children.only(children);

    // Info View mode: route the tip to the bottom bar; no popup.
    if (enabled) {
        return cloneElement(child, {
            onMouseEnter: (e) => { setHint(title); child.props?.onMouseEnter?.(e); },
            onMouseLeave: (e) => { setHint(null); child.props?.onMouseLeave?.(e); },
            onFocus: (e) => { setHint(title); child.props?.onFocus?.(e); },
            onBlur: (e) => { setHint(null); child.props?.onBlur?.(e); },
        });
    }

    const handleClick = (e) => {
        setOpen(false);
        // Preserve whatever the child already does on click.
        child.props?.onClick?.(e);
    };

    return (
        <MuiTooltip
            {...props}
            title={title}
            open={open}
            onOpen={() => setOpen(true)}
            onClose={() => setOpen(false)}
        >
            {cloneElement(child, { onClick: handleClick })}
        </MuiTooltip>
    );
}
