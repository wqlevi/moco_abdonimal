# =====> as class instance attributes
class Configuration(object):
    settings = None
    def __init__(self,k:str, v):
        self.settings = {k:v}

    @classmethod
    def update_setting(cls, key, value):
        v = cls(key, value)
        return v.settings


# Modifying class-level data using class methods
print(Configuration.update_setting('debug', True))


# =====> as class members
class Configuration(object):
    settings = None
    @classmethod
    def update_setting(cls, key, value):
        v = cls(key, value)
        return v.settings


# Modifying class-level data using class methods
print(Configuration.update_setting('debug', True))

