class IncidentMemory:

    def __init__(self):
        self.memory = []

    def add_incident(self, name, description, root_cause):

        self.memory.append({
            "name": name,
            "description": description,
            "root_cause": root_cause
        })

    def search(self, query):

        return [
            i for i in self.memory
            if query.lower() in i["description"].lower()
        ]