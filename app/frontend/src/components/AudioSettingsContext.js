import React, { createContext, useContext, useState } from 'react';

// The Performance engine owns the audio output; it publishes what the
// Settings dialog needs here: { supported, deviceId, devices, refresh, pick }.
const AudioSettingsContext = createContext({ api: null, setApi: () => {} });

export function AudioSettingsProvider({ children }) {
    const [api, setApi] = useState(null);
    return (
        <AudioSettingsContext.Provider value={{ api, setApi }}>
            {children}
        </AudioSettingsContext.Provider>
    );
}

export const useAudioSettings = () => useContext(AudioSettingsContext);
