"""CLI for the local context-layer and integration demo."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import dotenv_values
from openai import APIError, APIConnectionError

from context_layer import AccessDenied, DataQualityError, Gateway, Store, ask_openrouter

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "context.sqlite3")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--model")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create database and seed sample records without resetting existing data")
    for name in ("context", "ask", "propose"):
        command = commands.add_parser(name)
        command.add_argument("--actor", required=True)
        command.add_argument("--customer", required=True)
        command.add_argument("--ticket", required=True)
        if name == "ask":
            command.add_argument("--question", required=True)
        if name == "propose":
            command.add_argument("--action", required=True)
    decision = commands.add_parser("decide")
    decision.add_argument("--actor", required=True)
    decision.add_argument("--proposal", required=True, type=int)
    decision.add_argument("--decision", required=True, choices=["approve", "reject"])
    decision.add_argument("--reason", default="")
    audit = commands.add_parser("audit")
    audit.add_argument("--actor", required=True)
    audit.add_argument("--customer", required=True)
    review = commands.add_parser("pending", help="List proposals awaiting review")
    review.add_argument("--actor", required=True)
    review.add_argument("--customer", required=True)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    store = Store(args.db)
    gateway = Gateway(store)
    if args.command == "init":
        store.seed()
        print(f"Initialized fictional data in {args.db}")
    elif args.command == "context":
        print(json.dumps(gateway.context(args.actor, args.customer, args.ticket), indent=2))
    elif args.command == "ask":
        # Authorize before a model call or credential lookup.
        context = gateway.context(args.actor, args.customer, args.ticket)
        values = dotenv_values(args.env_file) if args.env_file.exists() else {}
        api_key = os.getenv("OPENROUTER_API_KEY") or values.get("OPENROUTER_API_KEY")
        model = args.model or os.getenv("OPENROUTER_MODEL") or values.get("OPENROUTER_MODEL") or "nvidia/nemotron-3.5-lightning:free"
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for ask")
        try:
            answer, usage = ask_openrouter(args.question, context, model, api_key)
        except (ValueError, APIError, APIConnectionError) as exc:
            store.audit(args.actor, "model.ask", args.customer, "error", type(exc).__name__)
            raise
        store.audit(args.actor, "model.ask", args.customer, "allowed", f"Model: {usage['model']}")
        print(json.dumps({"answer": answer, "model": usage["model"], "usage": usage["usage"],
                          "sources": [context["customer"]["provenance"], context["ticket"]["provenance"],
                                      context["playbook"], context["memory"]["provenance"]]}, indent=2))
    elif args.command == "propose":
        proposal_id = gateway.propose(args.actor, args.customer, args.ticket, args.action)
        print(f"Proposal {proposal_id} pending human review; no ticket update has been applied.")
    elif args.command == "decide":
        gateway.decide(args.actor, args.proposal, args.decision, args.reason)
        print(f"Proposal {args.proposal}: {args.decision}")
    elif args.command == "audit":
        print(json.dumps(gateway.audit_log(args.actor, args.customer), indent=2))
    elif args.command == "pending":
        print(json.dumps(gateway.pending_proposals(args.actor, args.customer), indent=2))


if __name__ == "__main__":
    try:
        main(parse_args())
    except (AccessDenied, DataQualityError, ValueError, OSError, sqlite3.Error, APIError, APIConnectionError) as exc:
        # Provider response bodies can contain sensitive upstream data.
        if isinstance(exc, (APIError, APIConnectionError)):
            print(f"error: OpenRouter request failed ({type(exc).__name__})", file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
