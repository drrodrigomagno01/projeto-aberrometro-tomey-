; Inno Setup — instalador do AberrometriaTMS (opcional; usado no GitHub Actions quando o ISCC está disponível)
; Compilar: iscc packaging\installer.iss  (após o PyInstaller gerar dist\AberrometriaTMS)
#define MyAppName "Aberrometria TMS (pesquisa)"
#define MyAppVersion "0.4.0"
#define MyAppExeName "AberrometriaTMS.exe"

[Setup]
AppId={{7E1D3C2A-5A2B-4C1E-9B6F-AB2F3E0C9D11}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Dr. Rodrigo Magno da Silva Oliveira — uso interno
DefaultDirName={autopf}\AberrometriaTMS
DefaultGroupName={#MyAppName}
OutputDir=..\dist
OutputBaseFilename=AberrometriaTMS-{#MyAppVersion}-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
WizardStyle=modern

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "Criar atalho na área de trabalho"; GroupDescription: "Atalhos:"

[Files]
Source: "..\dist\AberrometriaTMS\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Linha de comando (tmsab)"; Filename: "{cmd}"; Parameters: "/k ""{app}\tmsab.exe"" --help"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Abrir o programa"; Flags: nowait postinstall skipifsilent
