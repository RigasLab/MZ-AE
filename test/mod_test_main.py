import mod_test
import inspect

members = inspect.getmembers(mod_test)

class_members = {name: member for name, member in members if inspect.isclass(member)}

print(class_members)




