"""
Minimal Tkinter GUI: run with an audio file from the command line, view summary and reply, copy to clipboard.
The window opens only after processing; "Running..." is shown on the console. Errors are shown in the GUI.
Reply is copied to clipboard automatically when the window opens. Summary is displayed with markdown formatting.

Usage: summscriber-gui FILE
"""
import re
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from summscriber.cli import transcribe_audio, compute_summary_and_reply, get_fallback_summary
from summscriber.i18n import _


def _maximize_window(window: tk.Tk) -> None:
    """Maximize the window (cross-platform: Windows uses state, X11/Linux uses attributes)."""
    try:
        window.state("zoomed")
    except tk.TclError:
        try:
            window.attributes("-zoomed", True)
        except tk.TclError:
            pass


def _insert_markdown(text_widget: tk.Text, md: str) -> None:
    """Insert markdown string into a Text widget with bold, italic and heading tags. Widget must be NORMAL."""
    if not md:
        return
    text_widget.delete("1.0", tk.END)
    # Configure tags once
    text_widget.tag_configure("bold", font=("TkDefaultFont", 10, "bold"))
    text_widget.tag_configure("italic", font=("TkDefaultFont", 10, "italic"))
    text_widget.tag_configure("h1", font=("TkDefaultFont", 14, "bold"))
    text_widget.tag_configure("h2", font=("TkDefaultFont", 12, "bold"))
    text_widget.tag_configure("h3", font=("TkDefaultFont", 11, "bold"))

    lines = md.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### "):
            text_widget.insert(tk.END, stripped[4:] + "\n", "h3")
            continue
        if stripped.startswith("## "):
            text_widget.insert(tk.END, stripped[3:] + "\n", "h2")
            continue
        if stripped.startswith("# "):
            text_widget.insert(tk.END, stripped[2:] + "\n", "h1")
            continue
        # Inline **bold** and *italic*
        pos = 0
        while pos < len(line):
            # **bold**
            m = re.match(r"\*\*(.+?)\*\*", line[pos:])
            if m:
                start = text_widget.index(tk.END)
                text_widget.insert(tk.END, m.group(1), "bold")
                pos += len(m.group(0))
                continue
            # *italic*
            m = re.match(r"\*(.+?)\*", line[pos:])
            if m:
                text_widget.insert(tk.END, m.group(1), "italic")
                pos += len(m.group(0))
                continue
            # No match: advance one char or until next * or newline
            next_special = len(line)
            for i in range(pos, len(line)):
                if line[i] in "*\n":
                    next_special = i
                    break
            text_widget.insert(tk.END, line[pos:next_special])
            pos = next_special
        text_widget.insert(tk.END, "\n")
    # trim trailing newline
    if text_widget.get("1.0", tk.END).endswith("\n"):
        text_widget.delete("end-2c", tk.END)


def main() -> None:
    audio_path = (sys.argv[1:] or [None])[0]
    if not audio_path or not audio_path.strip():
        print(_("usage"), file=sys.stderr)
        sys.exit(1)

    print(_("running"), flush=True)
    try:
        trans = transcribe_audio(audio_path.strip())
        err = None
    except Exception as e:
        trans = None
        err = str(e)

    root = tk.Tk()
    root.title("Summscriber")
    root.minsize(450, 500)
    root.geometry("600x560")
    _maximize_window(root)

    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    main_frame.grid_rowconfigure(1, weight=2)
    main_frame.grid_rowconfigure(3, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)

    view_var = tk.StringVar(value="summary")
    summary_content = ""
    full_text_content = ""

    def _update_summary_view() -> None:
        summary_text.delete("1.0", tk.END)
        if view_var.get() == "summary":
            _insert_markdown(summary_text, summary_content)
        else:
            summary_text.insert(tk.END, full_text_content or "")

    radio_frame = ttk.Frame(main_frame)
    radio_frame.grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
    ttk.Radiobutton(
        radio_frame, text=_("summary"), variable=view_var, value="summary", command=_update_summary_view
    ).pack(side=tk.LEFT)
    ttk.Radiobutton(
        radio_frame, text=_("original_text"), variable=view_var, value="original", command=_update_summary_view
    ).pack(side=tk.LEFT, padx=(12, 0))

    summary_text = scrolledtext.ScrolledText(
        main_frame,
        height=8,
        wrap=tk.WORD,
        state=tk.NORMAL,
        font=("TkDefaultFont", 10),
        borderwidth=0,
        relief=tk.FLAT,
        highlightthickness=0,
        bg=root.cget("bg"),
        cursor="arrow",
        insertwidth=0,
    )
    summary_text.grid(row=1, column=0, sticky=tk.NSEW, pady=(12, 8), padx=(12, 12))
    summary_text.bind("<Key>", lambda e: "break")

    def _summary_context_menu(event: tk.Event) -> None:
        try:
            summary_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            has_selection = True
        except tk.TclError:
            has_selection = False
        menu = tk.Menu(summary_text, tearoff=0)
        if has_selection:
            menu.add_command(
                label=_("copy_selection"),
                command=lambda: _copy_selection_to_clipboard(summary_text, root),
            )
        menu.add_command(
            label=_("copy_all"),
            command=lambda: _copy_all_reply_to_clipboard(summary_text, root),
        )
        menu.tk_popup(event.x_root, event.y_root)

    summary_text.bind("<Button-3>", _summary_context_menu)

    reply_text = scrolledtext.ScrolledText(
        main_frame, height=6, wrap=tk.WORD, state=tk.NORMAL, font=("TkDefaultFont", 10)
    )

    def _reply_context_menu(event: tk.Event) -> None:
        try:
            reply_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            has_selection = True
        except tk.TclError:
            has_selection = False
        menu = tk.Menu(reply_text, tearoff=0)
        if has_selection:
            menu.add_command(
                label=_("copy_selection"),
                command=lambda: _copy_selection_to_clipboard(reply_text, root),
            )
        menu.add_command(
            label=_("copy_all"),
            command=lambda: _copy_all_reply_to_clipboard(reply_text, root),
        )
        menu.tk_popup(event.x_root, event.y_root)

    def _copy_selection_to_clipboard(widget: tk.Text, win: tk.Tk) -> None:
        try:
            sel = widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            if sel:
                win.clipboard_clear()
                win.clipboard_append(sel)
                win.update()
        except tk.TclError:
            pass

    def _copy_all_reply_to_clipboard(widget: tk.Text, win: tk.Tk) -> None:
        text = widget.get("1.0", tk.END).strip()
        if text:
            win.clipboard_clear()
            win.clipboard_append(text)
            win.update()

    reply_text.bind("<Button-3>", _reply_context_menu)

    def _close_on_enter(event: tk.Event | None = None) -> str | None:
        root.destroy()
        return "break"

    root.bind("<Return>", _close_on_enter)
    summary_text.bind("<Return>", lambda e: _close_on_enter(e))
    reply_text.bind("<Return>", lambda e: _close_on_enter(e))

    def _set_reply(content: str) -> None:
        reply_text.delete("1.0", tk.END)
        reply_text.insert(tk.END, content or "")

    ttk.Label(main_frame, text=_("reply")).grid(row=2, column=0, sticky=tk.W)
    reply_text.grid(row=3, column=0, sticky=tk.NSEW, pady=(0, 8))

    cancelled = [False]

    def _on_close() -> None:
        cancelled[0] = True
        root.destroy()

    def _apply_summary_reply(sr: dict) -> None:
        if cancelled[0]:
            return
        nonlocal summary_content
        summary_content = sr.get("summary", "")
        _update_summary_view()
        reply = sr.get("reply", "")
        _set_reply(reply)
        if reply:
            root.clipboard_clear()
            root.clipboard_append(reply)
            root.update()

    def _fetch_summary_reply() -> None:
        try:
            sr = compute_summary_and_reply(
                full_text_content,
                trans.get("language", ""),
                num_sentences=3,
                include_reply=True,
            )
            if cancelled[0]:
                return
            root.after(0, lambda: _apply_summary_reply(sr))
        except Exception as e:
            if cancelled[0]:
                return
            root.after(0, lambda: messagebox.showerror(_("error"), str(e), parent=root))

    root.protocol("WM_DELETE_WINDOW", _on_close)

    if err:
        summary_content = ""
        full_text_content = ""
        _update_summary_view()
        _set_reply("")
        messagebox.showerror(_("error"), err, parent=root)
    else:
        full_text_content = trans.get("text", "")
        summary_content = get_fallback_summary(
            full_text_content, trans.get("language", ""), num_sentences=3
        )
        if not summary_content:
            view_var.set("original")
        _update_summary_view()
        _set_reply(_("loading"))
        root.update_idletasks()
        root.update()
        threading.Thread(target=_fetch_summary_reply, daemon=True).start()

    root.mainloop()


if __name__ == "__main__":
    main()
