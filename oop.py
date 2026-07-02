class LibraryItem:
    """Base class for all library items."""

    def __init__(self, title, item_id):
        self.title = title
        self.item_id = item_id
        self._is_checked_out = False  # encapsulated state

    def get_details(self):
        return f"{self.title} (ID: {self.item_id})"

    def check_out(self):
        self._is_checked_out = True

    def return_item(self):
        self._is_checked_out = False

    @property
    def is_checked_out(self):
        return self._is_checked_out


class Book(LibraryItem):
    """Testing inheritance (is-a LibraryItem)."""

    def __init__(self, title, item_id, author, isbn):
        super().__init__(title, item_id)
        self.author = author
        self.isbn = isbn

    def get_details(self):
        return f"Book: {self.title} by {self.author}"


class DVD(LibraryItem):
    """Testing inheritance (is-a LibraryItem)."""

    def __init__(self, title, item_id, director, runtime_minutes):
        super().__init__(title, item_id)
        self.director = director
        self.runtime_minutes = runtime_minutes

    def get_details(self):
        return (
            f"DVD: {self.title} directed by {self.director}, "
            f"{self.runtime_minutes} minutes"
        )


class Member:
    """Represents a library member (has-a relationship with items)."""

    def __init__(self, name, member_id):
        self._name = name
        self._member_id = member_id
        self._borrowed_items = []

    def borrow_item(self, item):
        if item.is_checked_out:
            print(f"{item.title} is already checked out.")
            return

        self._borrowed_items.append(item)
        item.check_out()

    def return_item(self, item):
        if item in self._borrowed_items:
            self._borrowed_items.remove(item)
            item.return_item()

    def get_borrowed_items(self):
        return self._borrowed_items

    @property
    def name(self):
        return self._name


# -------------------------
# TESTING POLYMORPHISM
# -------------------------

b1 = Book("The Great Gatsby", "B001", "F. Scott Fitzgerald", "12345")
d1 = DVD("Inception", "D001", "Christopher Nolan", 148)

catalog = [b1, d1]

for item in catalog:
    print(item.get_details())


# -------------------------
# TESTING ENCAPSULATION
# -------------------------

member1 = Member("Alice", "M001")
member1.borrow_item(b1)

print(f"{member1.name} has borrowed:")
for item in member1.get_borrowed_items():
    print("-", item.title)