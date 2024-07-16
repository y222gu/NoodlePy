import tkinter as tk
from tkinter import ttk


class Root(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry('400x200')

        ttk.Style().element_create('Plain.Notebook.tab', 'from', 'default')
        ttk.Style().layout('TNotebook.Tab',
                           [('Plain.Notebook.tab', {'children':
                                [('Notebook.padding', {'side': 'top', 'children':
                                     [('Notebook.focus', {'side': 'top', 'children':
                                          [('Notebook.label', {'side': 'top', 'sticky': ''})],
                                      'sticky': 'nswe'})],
                                 'sticky': 'nswe'})],
                            'sticky': 'nswe'})])
        ttk.Style().configure('TNotebook', background='red', borderwidth=0)
        ttk.Style().configure('TNotebook.Tab', font=('Segoe UI', 14),
                              background='black', foreground='white', borderwidth=0)

        tab = ttk.Notebook(self)
        frm1 = tk.Frame(tab)
        tab.add(frm1, text='Frame1')
        frm2 = tk.Frame(tab)
        tab.add(frm2, text='Frame2')
        tab.pack(expand=True, fill='both')

        canv1 = tk.Canvas(frm1, bg='#0000FF')
        canv2 = tk.Canvas(frm2, bg='#00FF00')
        canv1.pack(expand=1, fill='both')
        canv2.pack(expand=1, fill='both')


if __name__ == '__main__':
    Root().mainloop()
    