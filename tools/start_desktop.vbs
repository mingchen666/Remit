' Remit developer desktop-shell launcher (Windows).
'
' Starts tools\desktop_app.py through the backend virtual environment without
' allocating a visible console window.
'
' Why this shim exists: uv creates the venv's pythonw.exe as a console-subsystem
' trampoline, so starting it directly allocates a console. When Windows Terminal
' is the default terminal application, Windows hands that console to Windows
' Terminal, which ignores the hidden flag and leaves an empty terminal window on
' the desktop. WScript.Shell.Run with a window style of 0 creates the console
' hidden, which keeps it a classic (invisible) conhost.
'
' Keep this file ASCII-only. wscript.exe reads .vbs files with the ANSI code page,
' so non-ASCII characters would be mangled in the message boxes below.

Option Explicit

Dim fso, shell, root, pythonw, app, command
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

root = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
app = fso.BuildPath(root, "tools\desktop_app.py")
pythonw = fso.BuildPath(root, "backend\.venv\Scripts\pythonw.exe")
If Not fso.FileExists(pythonw) Then
    pythonw = fso.BuildPath(root, "backend\venv\Scripts\pythonw.exe")
End If

If Not fso.FileExists(app) Then
    MsgBox "Desktop shell not found: " & app, 16, "Remit"
    WScript.Quit 1
End If

If Not fso.FileExists(pythonw) Then
    MsgBox "Backend virtual environment not found: " & vbCrLf & pythonw & vbCrLf & vbCrLf & _
        "Run: cd backend && uv sync", 16, "Remit"
    WScript.Quit 1
End If

shell.CurrentDirectory = root
command = """" & pythonw & """ """ & app & """"
shell.Run command, 0, False
