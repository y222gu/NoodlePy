"""Custom PyQt5 widgets for the spectrum viewer."""

from PyQt5.QtWidgets import QComboBox, QTableWidgetItem
from PyQt5.QtCore import Qt


class ScrollableComboBox(QComboBox):
    """
    QComboBox that lets you change the current item with the mouse wheel
    (without having to open the dropdown). We consume the wheel event so
    the parent scroll area won't scroll instead.
    """
    def wheelEvent(self, event):
        if self.count() == 0:
            event.ignore()
            return

        delta = event.angleDelta().y()

        if delta > 0:
            # scroll up -> previous item
            new_index = max(self.currentIndex() - 1, 0)
        elif delta < 0:
            # scroll down -> next item
            new_index = min(self.currentIndex() + 1, self.count() - 1)
        else:
            event.ignore()
            return

        if new_index != self.currentIndex():
            self.setCurrentIndex(new_index)

        # Don't let the event bubble up to the scroll area
        event.accept()


class SortableTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically when possible."""
    def __lt__(self, other):
        # If the other item isn't our type, fall back to default behavior
        if not isinstance(other, QTableWidgetItem):
            return super().__lt__(other)

        left_text = self.text()
        right_text = other.text()

        # Try numeric comparison first
        try:
            left_val = float(left_text)
            right_val = float(right_text)
            return left_val < right_val
        except ValueError:
            # Fallback: case-insensitive string comparison
            return left_text.lower() < right_text.lower()
