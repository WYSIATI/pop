"""Example: Structured Output with Pydantic
Requires: OPENAI_API_KEY environment variable (or appropriate provider key)
Requires: pydantic (`pip install pydantic`)

Demonstrates using Pydantic BaseModel as tool input for structured data.
The @tool decorator automatically generates JSON Schema from the model,
so the LLM knows exactly what shape of data to provide.
"""

from pydantic import BaseModel, Field

from pop import Agent, tool


# Define structured input with Pydantic
class ContactInfo(BaseModel):
    """Structured contact information."""

    name: str = Field(description="Full name of the person")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number in E.164 format")
    company: str = Field(description="Company or organization name")


@tool
def save_contact(contact: ContactInfo) -> str:
    """Save a contact to the database.

    Args:
        contact: The contact information to save.
    """
    # In production, this would persist to a real database.
    return f"Saved contact: {contact.name} ({contact.email}, {contact.phone}) at {contact.company}"


@tool
def list_contacts() -> str:
    """List all saved contacts."""
    return (
        "Current contacts:\n"
        "1. Alice Chen - alice@example.com - Acme Corp\n"
        "2. Bob Smith - bob@example.com - Widgets Inc"
    )


agent = Agent(
    model="openai:gpt-4o",
    tools=[save_contact, list_contacts],
    instructions="You are a CRM assistant. Help users manage their contacts.",
)

result = agent.run(
    "Add a new contact: Jane Doe, jane.doe@startup.io, +1-555-0199, works at TechStart AI"
)

print(result.output)
