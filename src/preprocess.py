import pandas as pd
import random
from datetime import datetime, timedelta


def generate_synthetic_incidents(n=5000):

    categories = ["Database", "Network", "Application", "Server", "Security"]

    root_cause_map = {
        "Database": "DB connection pool exhaustion",
        "Network": "Network latency",
        "Application": "NullPointerException",
        "Server": "Server overload",
        "Security": "Unauthorized access"
    }

    resolution_map = {
        "DB connection pool exhaustion": "Restart service, check connection pool settings",
        "Network latency": "Check routers, reduce traffic load",
        "NullPointerException": "Debug code, fix null checks",
        "Server overload": "Scale server, optimize processes",
        "Unauthorized access": "Reset credentials, check audit logs"
    }

    priorities = ["Low", "Medium", "High", "Critical"]
    services = ["UserService", "PaymentService", "AuthService", "DBService", "NetworkService"]

    start_time = datetime(2026, 1, 1)

    data = []

    for i in range(n):

        category = random.choice(categories)
        root_cause = root_cause_map[category]

        timestamp = start_time + timedelta(
            days=random.randint(0, 60),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        record = {
            "incident_id": i,
            "title": f"{category} issue #{i}",
            "description": f"{category} related failure observed in production",
            "category": category,
            "root_cause": root_cause,
            "resolution_steps": resolution_map[root_cause],
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "priority": random.choice(priorities),
            "service_name": random.choice(services)
        }

        data.append(record)

    return pd.DataFrame(data)

def preprocess(df):

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["text"] = (
        "Incident Description: " + df["description"] +
        ". Category: " + df["category"] +
        ". Root Cause: " + df["root_cause"] +
        ". Resolution: " + df["resolution_steps"] +
        ". Priority: " + df["priority"] +
        ". Service: " + df["service_name"]
    )

    return df