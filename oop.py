class LibraryItem():
    """This base class initializes the shared attributes."""
    def __init__(self, title, item_id, is_checked_out = False):
        self.title = title
        self.item_id = item_id
        self.is_checked_out = is_checked_out

    def get_details(self):
        return f"{self.title} (ID: {self.item_id})"
       

class Book(LibraryItem):
    """This class inherits the LibraryItem base (parent) class
    ['is a' relationship with the LibraryItem class]"""
    def __init__(self, title, item_id, author, isbn):
        super().__init__(title, item_id)
        self.author = author 
        self.isbn = isbn

    def get_details(self):   # This inherited method is here overriden to show polymorphism
        return f"Book: {self.title} by {self.author}"


class DVD(LibraryItem):
    """This class inherits the LibraryItem base (parent) class
    ['is a' relationship with the LibraryItem class]"""
    def __init__(self, title, item_id, director, runtime_minutes):
        super().__init__( title, item_id)
        self.director = director
        self.runtime_minutes = runtime_minutes

    def get_details(self):   # This inherited method is here overriden to show polymorphism
        return(f"{self.title} was directed by {self.director} and runs for {self.runtime_minutes} minutes!")

class Member(): 
    """This class demonstrates composition instead of inheritance 
    ["has a" relationship with Book and DVD classes]"""
    def __init__(self, name, member_id):
        """ A private attributes to demonstrate encapsulation n Python"""
        self._name = name              
        self._member_id = member_id  
        self._borrowed_items = []

    def borrow_item(self, item):
        self._borrowed_items.append(item)
        item.is_checked_out = True

    @property         # This decorator allows "read-only" access to the private attribute self._name
    def name(self):
        return self._name
        



# Create objects
b1 = Book("The Great Gatsby", "B001", "F. Scott Fitzgerald", "12345")
d1 = DVD("Inception", "D001", "Christopher Nolan", 148)

# Test Polymorphism
catalog = [b1, d1]

for item in catalog:
    # This calls the specific get_details for each type!
    print(item.get_details())

# Test Member/Encapsulation
member1 = Member("Alice", "M001")
member1.borrow_item(b1)
print(f"{member1.name} has borrowed: {member1._borrowed_items[0].title}")


