"""Connectors — Execution Layer.

A ``Connector`` is a declared integration into a real external system — an EHR,
a lab system, an ERP, a messaging channel. It records *what* a system is and
*whether it can perform side-effecting actions* (``write``), so the Safety Layer
can gate writes behind approval. Auth is declared, never stored on the connector
(it belongs in a vault), mirroring the agent-card / MCP model.

The bundled :data:`CONNECTORS` registry ships 50+ pre-built connectors across
clinical and operational categories — the "connect existing systems without
replacing them" promise. Each connector can be bridged into the Intelligence
Layer as an :class:`~erdos_fai.intelligence.mcp.MCPServer` via :meth:`Connector.to_mcp`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..intelligence.mcp import MCPServer

# Connector categories — the surfaces a healthcare/operations org needs to touch.
CATEGORIES = (
    "EHR",
    "Interoperability",
    "Labs & Diagnostics",
    "Pharmacy",
    "Scheduling & Capacity",
    "Messaging & Notifications",
    "ERP & Finance",
    "Payments & Claims",
    "Identity & Access",
    "Data & Storage",
    "Analytics & BI",
    "Forms & Intake",
    "Generic",
)


@dataclass(frozen=True)
class Connector:
    """A declared integration into an external system."""

    id: str
    name: str
    category: str
    description: str = ""
    write: bool = False           # True => can perform side-effecting actions (Safety-gated)
    auth: str = "oauth2"          # how it authenticates; secrets live in a vault, not here
    standard: str = ""            # interoperability standard, if any (FHIR, HL7v2, DICOM…)

    def card(self) -> str:
        flag = " (write)" if self.write else ""
        std = f" · {self.standard}" if self.standard else ""
        return f"- `{self.id}`{flag} — {self.name} [{self.category}{std}]: {self.description}"

    def to_mcp(self, url: str | None = None) -> MCPServer:
        """Bridge this connector into the Intelligence Layer as an MCP server."""
        return MCPServer(
            name=self.id,
            url=url or f"mcp://connectors/{self.id}",
            description=f"{self.name} — {self.description}",
        )


class ConnectorRegistry:
    """An addressable catalog of connectors."""

    def __init__(self, connectors: list[Connector] | None = None) -> None:
        self._by_id: dict[str, Connector] = {}
        for c in connectors or []:
            self.register(c)

    def register(self, connector: Connector) -> Connector:
        self._by_id[connector.id] = connector
        return connector

    def get(self, connector_id: str) -> Connector:
        return self._by_id[connector_id]

    def all(self) -> list[Connector]:
        return list(self._by_id.values())

    def categories(self) -> list[str]:
        seen: list[str] = []
        for c in self._by_id.values():
            if c.category not in seen:
                seen.append(c.category)
        return seen

    def by_category(self, category: str) -> list[Connector]:
        return [c for c in self._by_id.values() if c.category == category]

    def writable(self) -> list[Connector]:
        return [c for c in self._by_id.values() if c.write]

    def catalog(self) -> str:
        return "\n".join(c.card() for c in self._by_id.values())

    def __contains__(self, connector_id: str) -> bool:
        return connector_id in self._by_id

    def __iter__(self):
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)


# (id, name, category, write, description, standard)
_BUILTIN: list[tuple[str, str, str, bool, str, str]] = [
    # EHR
    ("epic", "Epic", "EHR", True, "Read charts and write orders, notes, and flags via the Epic API.", "FHIR"),
    ("oracle-health", "Oracle Health (Cerner)", "EHR", True, "Clinical data exchange and order write-back.", "FHIR"),
    ("meditech", "MEDITECH Expanse", "EHR", True, "Inpatient charting and results access.", "FHIR"),
    ("veradigm", "Veradigm (Allscripts)", "EHR", True, "Ambulatory EHR data and document write-back.", "FHIR"),
    ("athenahealth", "athenahealth", "EHR", True, "Practice, billing, and clinical records.", "FHIR"),
    ("nextgen", "NextGen Healthcare", "EHR", True, "Ambulatory charting and tasks.", "FHIR"),
    ("eclinicalworks", "eClinicalWorks", "EHR", True, "Clinical documentation and messaging.", "FHIR"),
    ("greenway", "Greenway Health", "EHR", False, "Read clinical and practice data.", "FHIR"),
    # Interoperability
    ("fhir", "HL7 FHIR R4", "Interoperability", True, "Standards-native read/write to any FHIR endpoint.", "FHIR R4"),
    ("hl7v2", "HL7 v2 Interface", "Interoperability", True, "ADT, ORM, ORU message ingest and emit.", "HL7v2"),
    ("smart-on-fhir", "SMART on FHIR", "Interoperability", True, "Scoped, app-launch authorization over FHIR.", "FHIR"),
    ("ccda", "C-CDA Documents", "Interoperability", False, "Parse consolidated clinical document architecture.", "C-CDA"),
    ("redox", "Redox", "Interoperability", True, "Unified integration engine across EHR networks.", "FHIR"),
    ("1up-health", "1upHealth", "Interoperability", False, "Aggregated patient FHIR data.", "FHIR"),
    # Labs & Diagnostics
    ("labcorp", "Labcorp", "Labs & Diagnostics", False, "Order status and result retrieval.", "HL7v2"),
    ("quest", "Quest Diagnostics", "Labs & Diagnostics", False, "Lab orders and structured results.", "HL7v2"),
    ("lis", "LIS (generic)", "Labs & Diagnostics", True, "Laboratory information system results and flags.", "HL7v2"),
    ("pacs-dicom", "PACS / DICOM", "Labs & Diagnostics", False, "Imaging study metadata and availability.", "DICOM"),
    ("ris", "Radiology RIS", "Labs & Diagnostics", True, "Radiology orders and report status.", "HL7v2"),
    # Pharmacy
    ("surescripts", "Surescripts", "Pharmacy", True, "E-prescribing and medication history.", "NCPDP"),
    ("rxnorm", "RxNorm", "Pharmacy", False, "Normalized drug nomenclature lookups.", ""),
    # Scheduling & Capacity
    ("epic-cadence", "Epic Cadence", "Scheduling & Capacity", True, "Appointment slots and scheduling.", "FHIR"),
    ("qgenda", "QGenda", "Scheduling & Capacity", True, "Provider and staff scheduling.", ""),
    ("outlook-calendar", "Microsoft Outlook Calendar", "Scheduling & Capacity", True, "Calendar events and availability.", ""),
    ("google-calendar", "Google Calendar", "Scheduling & Capacity", True, "Event creation and free/busy.", ""),
    ("teamup", "Teamup", "Scheduling & Capacity", True, "Shared resource and bed scheduling.", ""),
    # Messaging & Notifications
    ("twilio-sms", "Twilio SMS", "Messaging & Notifications", True, "Send SMS and voice notifications.", ""),
    ("whatsapp", "WhatsApp Business", "Messaging & Notifications", True, "Send templated WhatsApp messages.", ""),
    ("slack", "Slack", "Messaging & Notifications", True, "Post to channels and DMs.", ""),
    ("teams", "Microsoft Teams", "Messaging & Notifications", True, "Channel and chat notifications.", ""),
    ("sendgrid", "SendGrid", "Messaging & Notifications", True, "Transactional email delivery.", ""),
    ("outlook-mail", "Microsoft Outlook Mail", "Messaging & Notifications", True, "Send and read mailbox messages.", ""),
    ("pagerduty", "PagerDuty", "Messaging & Notifications", True, "On-call paging and incident escalation.", ""),
    # ERP & Finance
    ("sap", "SAP S/4HANA", "ERP & Finance", True, "Procurement, inventory, and finance.", ""),
    ("oracle-erp", "Oracle ERP Cloud", "ERP & Finance", True, "Purchasing and financials.", ""),
    ("workday", "Workday", "ERP & Finance", True, "HR, staffing, and finance records.", ""),
    ("netsuite", "NetSuite", "ERP & Finance", True, "ERP and accounts management.", ""),
    ("coupa", "Coupa", "ERP & Finance", True, "Procurement and spend approvals.", ""),
    # Payments & Claims
    ("stripe", "Stripe", "Payments & Claims", True, "Payments and payouts.", ""),
    ("change-healthcare", "Change Healthcare", "Payments & Claims", True, "Claims clearinghouse and eligibility.", "X12"),
    ("availity", "Availity", "Payments & Claims", True, "Payer connectivity and authorizations.", "X12"),
    ("waystar", "Waystar", "Payments & Claims", True, "Revenue-cycle and claims management.", "X12"),
    # Identity & Access
    ("okta", "Okta", "Identity & Access", False, "SSO, users, and group membership.", "SAML/OIDC"),
    ("entra-id", "Microsoft Entra ID", "Identity & Access", False, "Directory, SSO, and roles.", "SAML/OIDC"),
    ("auth0", "Auth0", "Identity & Access", False, "Authentication and user profiles.", "OIDC"),
    # Data & Storage
    ("s3", "Amazon S3", "Data & Storage", True, "Object storage read/write.", ""),
    ("gcs", "Google Cloud Storage", "Data & Storage", True, "Object storage read/write.", ""),
    ("azure-blob", "Azure Blob Storage", "Data & Storage", True, "Object storage read/write.", ""),
    ("snowflake", "Snowflake", "Data & Storage", False, "Warehouse queries for analytics.", "SQL"),
    ("postgres", "PostgreSQL", "Data & Storage", True, "Relational reads and writes.", "SQL"),
    ("sftp", "SFTP", "Data & Storage", True, "Batch file exchange.", ""),
    # Analytics & BI
    ("tableau", "Tableau", "Analytics & BI", False, "Dashboards and published data sources.", ""),
    ("power-bi", "Microsoft Power BI", "Analytics & BI", False, "Reports and datasets.", ""),
    ("looker", "Looker", "Analytics & BI", False, "Modeled metrics and explores.", ""),
    # Forms & Intake
    ("docusign", "DocuSign", "Forms & Intake", True, "E-signature envelopes and status.", ""),
    ("typeform", "Typeform", "Forms & Intake", False, "Intake form submissions.", ""),
    ("jotform", "Jotform", "Forms & Intake", False, "Form responses and uploads.", ""),
    # Generic
    ("rest-webhook", "REST / Webhook", "Generic", True, "Call any HTTP endpoint or receive webhooks.", "HTTP"),
    ("mcp-server", "MCP Server", "Generic", True, "Any Model Context Protocol server.", "MCP"),
    ("graphql", "GraphQL", "Generic", True, "Query and mutate any GraphQL API.", "HTTP"),
]

CONNECTORS = ConnectorRegistry(
    [
        Connector(id=i, name=n, category=c, write=w, description=d, standard=s)
        for (i, n, c, w, d, s) in _BUILTIN
    ]
)
