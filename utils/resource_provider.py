class ExternalResourceProvider:

    RESOURCE_MAPPINGS = {
        "python": ["Real Python", "Python Docs"],
        "machine learning": ["Coursera ML", "Kaggle Learn"],
    }

    @classmethod
    def suggest_resources(cls, keywords):
        suggestions = {}

        for k in keywords:
            for topic, res in cls.RESOURCE_MAPPINGS.items():
                if k in topic:
                    suggestions[k] = res

        if not suggestions:
            suggestions["general"] = [
                "Google Search",
                "Stack Overflow"
            ]

        return suggestions