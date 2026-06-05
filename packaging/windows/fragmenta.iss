; Inno Setup script for Fragmenta (Windows).
;
; Bundles the frozen launcher (packaging/build/win/fragmenta/) + the assembled
; payload (packaging/build/payload/, incl. python-3.11) into a per-user
; installer that needs no admin rights.
;
; Build (from repo root, after assemble.py + pyinstaller):
;   iscc /DAppVersion=1.0.0 packaging\windows\fragmenta.iss
;
; Output: packaging\build\Fragmenta-<version>-Setup.exe

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{B3A6F0C2-7E2E-4E2A-9E1F-FRAGMENTA0002}
AppName=Fragmenta
AppVersion={#AppVersion}
AppPublisher=Misagh Azimi
DefaultDirName={localappdata}\Programs\Fragmenta
DefaultGroupName=Fragmenta
DisableProgramGroupPage=yes
; Per-user install — no admin/UAC, avoids Program Files read-only issues.
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\build
OutputBaseFilename=Fragmenta-{#AppVersion}-Setup
SetupIconFile=..\..\app\frontend\public\fragmenta.ico
UninstallDisplayIcon={app}\fragmenta.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
; Frozen launcher (fragmenta.exe + _internal).
Source: "..\build\win\fragmenta\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
; Assembled payload: app code + standalone python-3.11. Sits beside the exe so
; launcher._payload_dir() resolves to {app}.
Source: "..\build\payload\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Fragmenta"; Filename: "{app}\fragmenta.exe"
Name: "{group}\Uninstall Fragmenta"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Fragmenta"; Filename: "{app}\fragmenta.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\fragmenta.exe"; Description: "Launch Fragmenta"; Flags: nowait postinstall skipifsilent

; Remove the venv + caches created at runtime under %APPDATA% on uninstall is
; intentionally NOT done here — user data (models, projects, output) lives in
; %APPDATA%\FragmentaDesktop and is preserved. Document this for users.

[Code]
function WebView2Installed(): Boolean;
var
  v: String;
begin
  // Evergreen WebView2 runtime registers a version under these keys.
  Result :=
    RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', v) or
    RegQueryStringValue(HKCU, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', v);
end;

procedure EnsureWebView2();
var
  tmp, url: String;
  code: Integer;
begin
  if WebView2Installed() then
    exit;
  if MsgBox('Fragmenta needs the Microsoft Edge WebView2 runtime, which is not installed. Download and install it now?',
            mbConfirmation, MB_YESNO) = IDYES then
  begin
    url := 'https://go.microsoft.com/fwlink/p/?LinkId=2124703'; { Evergreen bootstrapper }
    tmp := ExpandConstant('{tmp}\MicrosoftEdgeWebview2Setup.exe');
    if DownloadTemporaryFile(url, 'MicrosoftEdgeWebview2Setup.exe', '', nil) <> '' then
    begin
      Exec(tmp, '/silent /install', '', SW_SHOW, ewWaitUntilTerminated, code);
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    EnsureWebView2();
end;
