import tkinter as tk
from tkinter import ttk

class Subscriber:
    def __init__(self):
        print('Subscriber inited')

    def update(self, message):
        print('{} got message "{}"'.format(self.name, message))
        
class Publisher:
    def __init__(self, events):
        # Maps event names to subscribers
        self.events = {event: dict() for event in events}
    def get_subscribers(self, event):
        return self.events[event]
    def register(self, event, who, callback=None):
        if callback is None:
            callback = getattr(who, 'update')
        self.get_subscribers(event)[who] = callback
    def unregister(self, event, who):
        del self.get_subscribers(event)[who]
    def dispatch(self, event, message):
        for subscriber, callback in self.get_subscribers(event).items():
            # print('Distpatching event "{}" to {}'.format(event, subscriber))
            callback(message)


class GUI_with_buttons(Publisher, ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)  # Initialize ttk.Frame first
        Publisher.__init__(self, ['event_1', 'event_2'])
        
        self.switch_button = ttk.Button(self, text='Switch', command=self.switch)
        self.switch_button.pack()  # Add switch button to the frame

        # # a dfifferent button to trigger another event
        # self.another_event = ttk.Button(self, text='Another Event', command=self.another_event)
        # self.another_event.pack()

    def switch(self):
        if self.switch_button['text'] == 'Switch_1':
            self.switch_button['text'] = 'Unswitch_1'
            self.dispatch('event_1', 'first button was turned off')
        else:
            self.switch_button['text'] = 'Switch_1'
            self.dispatch('event_2', 'first button was turned on')

# a ttk.Frame a text window display the message that triggered by the event
class A(ttk.Frame, Subscriber):
    def __init__(self):
        super().__init__()
        Subscriber.__init__(self)
        self.name = 'A'
        self.text = tk.Text(self, width=40, height=10)
        self.text.pack()

    def handle_event_1(self, message):
        self.text.insert(tk.END, 'Handling event 1, A got message "{}"\n'.format(message))
        self.text.see(tk.END)

    def handle_event_2(self, message):
        self.text.insert(tk.END, 'Handling event 2, A got message "{}"\n'.format(message))
        self.text.see(tk.END)

class B(Subscriber):
    def __init__(self):
        super().__init__()
    def handle_event_1(self, message):
        print('Handling event 1, B got message "{}"'.format(message))

    def handle_event_2(self, message):
        print('Handling event 2, B got message "{}"'.format(message))

if __name__ == '__main__':
    root = tk.Tk()

    app = GUI_with_buttons(root)
    app.pack()

    a = A() 
    a.pack()
    app.register('event_1', a, a.handle_event_1)
    app.register('event_2', a, a.handle_event_2)

    b = B()
    app.register('event_1', b, b.handle_event_1)
    app.register('event_2', b, b.handle_event_2)
    
    root.mainloop()
