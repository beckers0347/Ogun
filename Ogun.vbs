' Ogun.vbs - start Ogun in its own window, with no console.
' Double-click this file (or a shortcut to it). It runs run_ogun.bat --ui hidden; the first start also sets up .venv,
' so it can take a minute before the window appears. If the window never appears, run run_ogun.bat --ui in a terminal
' once to see the error. The log is in %LOCALAPPDATA%\Ogun\ogun.log.
Option Explicit
Dim fso, sh, here, q
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
here = fso.GetParentFolderName(WScript.ScriptFullName)
q = Chr(34)
sh.CurrentDirectory = here
' window style 0 = hidden; False = don't wait
sh.Run "cmd /c " & q & q & here & "\run_ogun.bat" & q & " --ui" & q, 0, False
