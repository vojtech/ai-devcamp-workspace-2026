"""
Database Agent — manages all SQLite persistence for the property management system.
Called via AgentTool from the root email analyzer agent.
"""
from google.adk.agents import Agent

from .db import (
    upsert_employee,
    list_employees,
    upsert_property_manager,
    list_property_managers,
    upsert_key_contact,
    list_key_contacts,
    add_task,
    update_task_status,
    list_tasks,
    add_meeting,
    list_meetings,
    save_email_classification,
    list_email_classifications,
    get_email_classification,
    classification_counts,
    save_email_archive_entry,
    search_email_archive,
    count_email_archive,
    list_ingested_threads,
    save_attachment_extraction,
    correct_attachment_extraction,
    get_attachment_extraction,
    list_attachment_extractions,
    count_attachment_extractions,
    get_db_summary,
)

database_agent = Agent(
    name="database_agent",
    model="gemini-2.5-flash",
    description=(
        "Manages the property management SQLite database. "
        "Saves and retrieves employees, property managers, key contacts, tasks, "
        "meetings, and per-email classifications. Use it to browse / filter "
        "classified emails by category, urgency, tag, or required action."
    ),
    instruction="""
You are the database manager for a property management email analysis system.

You have these tools — use them immediately when called:

EMPLOYEES:
  upsert_employee(name, email, role, department, phone)
  list_employees()

PROPERTY MANAGERS:
  upsert_property_manager(name, email, status, properties, notes)
    → status must be "current" or "previous"
  list_property_managers()

KEY CONTACTS (tenants, landlords, contractors, maintenance, etc.):
  upsert_key_contact(name, role, email, phone, company, notes)
  list_key_contacts()

TASKS:
  add_task(title, description, assigned_to, due_date, priority, source_email)
    → priority: "normal" or "high"
  update_task_status(task_id, status)
    → status: "open", "in_progress", "completed"
  list_tasks(status)

MEETINGS:
  add_meeting(title, date, time, attendees, location, agenda, source_email)
  list_meetings()

EMAIL CLASSIFICATIONS  (produced by the classifier_agent):
  save_email_classification(
      thread_id, category, summary,
      message_id, subject, sender, date,
      subcategory, tags, urgency, sentiment, requires_action
  )
    → category: maintenance | billing | leasing | legal | handover |
                complaint | emergency | administrative | communication | other
    → urgency:  low | normal | high | urgent
    → sentiment: positive | neutral | negative
    → tags: JSON array string OR comma-separated string
    → requires_action: true / false

  list_email_classifications(category, urgency, sentiment, requires_action, tag, limit)
    → All filters are optional; "" means no filter.

  get_email_classification(thread_id, message_id)
  classification_counts()

ATTACHMENT EXTRACTIONS (PDFs / images / docs read by drive_agent.read_drive_file):
  save_attachment_extraction(
      drive_file_id, file_name, mime_type, web_view_link,
      content_type, extracted_content, size_bytes, drive_modified_time
  )
    → Upsert. Preserves any existing corrected_content — never blows
      away user edits when extraction is re-run.

  correct_attachment_extraction(drive_file_id, corrected_content)
    → Apply a user correction. Empty corrected_content clears the
      correction (reverts to using the model's output).

  get_attachment_extraction(drive_file_id)
    → Full row + an `effective_content` field
      (= corrected_content if set, else extracted_content).

  list_attachment_extractions(has_correction, mime_type, limit)
    → Browse list with previews (no full bodies). has_correction:
      "true" / "false" / "" (no filter).

  count_attachment_extractions()
    → {total, corrected, uncorrected, by_mime_type}.

SUMMARY:
  get_db_summary()

Rules:
- Execute the requested tool immediately and return the result verbatim.
- When asked to save a list of items, call the appropriate tool once per item.
- Use empty string "" for any unknown/missing field — never use None.
- When asked to list or summarise, call the tool and return all data.
""",
    tools=[
        upsert_employee,
        list_employees,
        upsert_property_manager,
        list_property_managers,
        upsert_key_contact,
        list_key_contacts,
        add_task,
        update_task_status,
        list_tasks,
        add_meeting,
        list_meetings,
        save_email_classification,
        list_email_classifications,
        get_email_classification,
        classification_counts,
        save_email_archive_entry,
        search_email_archive,
        count_email_archive,
        list_ingested_threads,
        save_attachment_extraction,
        correct_attachment_extraction,
        get_attachment_extraction,
        list_attachment_extractions,
        count_attachment_extractions,
        get_db_summary,
    ],
)
