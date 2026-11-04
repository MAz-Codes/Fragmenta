; Inno Setup script for Fragmenta (Windows).
;
; Bundles the frozen launcher (packaging/build/win/fragmenta/) + the assembled
; bundle (packaging/build/bundle/: runtime\ + payload\) into a per-user
; installer that needs no admin rights.
;
; Build (from repo root, after assemble.py + pyinstaller; build_exe.ps1 does all three):
;   iscc /DAppVersion=2.0.0 /DVersionInfo=2.0.0.100 packaging\windows\fragmenta.iss
;
; Output: packaging\build\Fragmenta-<version>-Windows-x64-Setup.exe
;
; User data (models, projects, output, the ~8 GB venv) lives in
; %LOCALAPPDATA%\FragmentaDesktop, outside {app}. Uninstall keeps it, and offers
; to remove only the venv and download cache, which the app can rebuild.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef VersionInfo
  #define VersionInfo "0.0.0.0"
#endif

[Setup]
; Not a valid GUID, but Inno only uses it as a name. Never change it: upgrades
; find the installed copy by it.
AppId={{B3A6F0C2-7E2E-4E2A-9E1F-FRAGMENTA0002}
AppName=Fragmenta
AppVersion={#AppVersion}
VersionInfoVersion={#VersionInfo}
AppPublisher=Misagh Azimi
AppPublisherURL=https://github.com/MAz-Codes/Fragmenta
DefaultDirName={localappdata}\Programs\Fragmenta
DefaultGroupName=Fragmenta
DisableProgramGroupPage=yes
; Per-user install — no admin/UAC, avoids Program Files read-only issues.
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir=..\build
OutputBaseFilename=Fragmenta-{#AppVersion}-Windows-x64-Setup
SetupIconFile=..\..\app\frontend\public\fragmenta.ico
UninstallDisplayIcon={app}\fragmenta.exe
UninstallDisplayName=Fragmenta
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The launcher holds this mutex while Fragmenta runs: Setup asks to close it
; first, then closes whatever still uses files in {app} (the app's Python).
AppMutex=FragmentaLauncherMutex
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[InstallDelete]
; An upgrade replaces these whole, so files dropped since the last version
; can't linger and shadow new code.
Type: filesandordirs; Name: "{app}\payload"
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\_internal"

[Files]
; Frozen launcher (fragmenta.exe + _internal).
Source: "..\build\win\fragmenta\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
; Assembled bundle: runtime\ (Python + uv) and payload\ (app code). Sits
; beside the exe, so launcher._bundle_root() resolves to {app}.
Source: "..\build\bundle\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[UninstallDelete]
; Python writes __pycache__ folders here at run time.
Type: filesandordirs; Name: "{app}\payload"
Type: filesandordirs; Name: "{app}\runtime"

[Icons]
Name: "{group}\Fragmenta"; Filename: "{app}\fragmenta.exe"
Name: "{group}\Uninstall Fragmenta"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Fragmenta"; Filename: "{app}\fragmenta.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\fragmenta.exe"; Description: "Launch Fragmenta"; Flags: nowait postinstall skipifsilent

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

// PyTorch needs the Microsoft C++ runtime, and its Windows build doesn't carry
// msvcp140.dll. Most PCs already have it from another app.
function VCRuntimeInstalled(): Boolean;
begin
  Result := FileExists(ExpandConstant('{sys}\msvcp140.dll')) and
            FileExists(ExpandConstant('{sys}\vcruntime140_1.dll'));
end;

// Download a Microsoft installer and run it. Errors are shown, not fatal:
// Fragmenta itself is installed either way.
procedure RunDownloaded(const Url, FileName, Params, Verb: String);
var
  code: Integer;
begin
  try
    DownloadTemporaryFile(Url, FileName, '', nil);
    if not ShellExec(Verb, ExpandConstant('{tmp}\' + FileName), Params, '', SW_SHOW, ewWaitUntilTerminated, code) then
      SuppressibleMsgBox(FileName + ' did not start: ' + SysErrorMessage(code), mbError, MB_OK, IDOK);
  except
    SuppressibleMsgBox('Could not download ' + FileName + ': ' + GetExceptionMessage, mbError, MB_OK, IDOK);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep <> ssPostInstall then
    exit;
  if not WebView2Installed() and
     (SuppressibleMsgBox('Fragmenta needs the Microsoft Edge WebView2 runtime, which is not installed. Download and install it now?',
                         mbConfirmation, MB_YESNO, IDNO) = IDYES) then
    RunDownloaded('https://go.microsoft.com/fwlink/p/?LinkId=2124703', 'MicrosoftEdgeWebview2Setup.exe',
                  '/silent /install', '');
  // The C++ runtime installs for the whole PC, so Windows asks for permission once.
  if not VCRuntimeInstalled() and
     (SuppressibleMsgBox('Fragmenta needs the Microsoft Visual C++ runtime, which is not installed. Download and install it now? Windows will ask for permission.',
                         mbConfirmation, MB_YESNO, IDNO) = IDYES) then
    RunDownloaded('https://aka.ms/vs/17/release/vc_redist.x64.exe', 'vc_redist.x64.exe',
                  '/install /passive /norestart', 'runas');
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  data: String;
begin
  if CurUninstallStep <> usPostUninstall then
    exit;
  data := ExpandConstant('{localappdata}\FragmentaDesktop');
  if not DirExists(data + '\venv') then
    exit;
  if SuppressibleMsgBox('Also remove the Python packages Fragmenta downloaded (about 8 GB)?' + #13#10#13#10 +
                        'Your models, projects and outputs are kept in ' + data + '.',
                        mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
  begin
    DelTree(data + '\venv', True, True, True);
    DelTree(data + '\cache', True, True, True);
  end;
end;
