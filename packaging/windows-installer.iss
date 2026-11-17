; Inno Setup script — builds a real Windows installer around the one-file exe.
; Requires: dist\PHS-Studio.exe already built by scripts/package_release.py
; Build:    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\windows-installer.iss

#define AppName "PHS Studio"
#define AppVersion "1.0.0"
#define AppPublisher "PHS Team"
#define AppExe "PHS-Studio.exe"

[Setup]
AppId={{8E4A1F52-6C7B-4B0E-9A31-5D8F2C7B1E44}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\PHS Studio
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=PHS-Studio-{#AppVersion}-windows-installer
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
; ChineseSimplified.isl is not shipped with Inno Setup, so the wizard is English.
; Drop a copy into <Inno>\Languages and uncomment to localise.
; Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\PHS-Studio.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\卸载 PHS Studio"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "启动 {#AppName}"; Flags: nowait postinstall skipifsilent
