import React, { cloneElement, useState } from 'react';
import { Tooltip as MuiTooltip } from '@mui/material';

/**
 * App-wide Tooltip.
 *
 * Thin wrapper around MUI's Tooltip that additionally dismisses the tooltip
 * the instant its child is activated (clicked / key-pressed). Stock MUI keeps
 * the tooltip open after a click because the control retains hover + focus,
 * so help text lingers over a button you've already pressed. We take control
 * of `open` and force it shut on click, then let the normal hover/focus
 * listeners reopen it on the next interaction.
 *
 * The API is identical to MUI's Tooltip — every prop (title, placement,
 * arrow, enterDelay, …) passes straight through, and the child's own onClick
 * is preserved. Drop-in replacement for `import { Tooltip } from '@mui/material'`.
 */
export default function Tooltip({ children, ...props }) {
    const [open, setOpen] = useState(false);
    const child = React.Children.only(children);

    const handleClick = (e) => {
        setOpen(false);
        // Preserve whatever the child already does on click.
        child.props?.onClick?.(e);
    };

    return (
        <MuiTooltip
            {...props}
            open={open}
            onOpen={() => setOpen(true)}
            onClose={() => setOpen(false)}
        >
            {cloneElement(child, { onClick: handleClick })}
        </MuiTooltip>
    );
}
