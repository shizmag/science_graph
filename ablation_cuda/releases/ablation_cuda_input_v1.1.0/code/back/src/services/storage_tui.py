"""
Storage TUI — interactive console browser for papers, authors, and concepts.
"""

import click
import math
import json
import os
import sys
import webbrowser
import subprocess
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.panel import Panel
from rich import box

from src import console as con
from src.prompts import prompts


def run_storage_tui(graph_repo: Any, container: Any, limit: int = 20) -> None:
    """Runs the interactive console storage browser using click."""
    TABLES = ["documents", "authors", "concepts"]
    TABLE_LABELS = {"documents": "📚 Documents", "authors": "👥 Authors", "concepts": "🧠 Concepts"}

    page = 1
    active_table = "documents"   # currently focused table
    selected_idx = None          # 1-based row number within the current page
    status_msg = ""              # feedback line after actions
    search_query = None          # search term

    def _get_rows(tbl: str, pg: int):
        return graph_repo.get_browse_rows(tbl, pg, limit, search_query)

    def _count(tbl: str) -> int:
        return graph_repo.get_browse_count(tbl, search_query)

    def _delete_node(node_id: str) -> str:
        try:
            graph_repo.delete_node(node_id)
            return f"[bold red]✗  Deleted[/bold red] node [dim]{node_id[:40]}[/dim]"
        except Exception as exc:
            return f"[red]Error deleting: {exc}[/red]"

    def _edit_node(node_id: str, tbl: str) -> str:
        """Interactive field editor — prompts for new values for key fields."""
        node = graph_repo.get_node_by_id(node_id)
        if not node:
            return "[red]Record not found.[/red]"
        label, props_str = node
        props = json.loads(props_str)
        con.console.print("\n[bold yellow]— Edit Record —[/bold yellow] (Enter = keep current, Ctrl+C = cancel)\n")
        try:
            if tbl == "documents":
                new_title = input(f"  Title [{props.get('title', '')}]: ").strip()
                if new_title:
                    props["title"] = new_title
                raw_authors = props.get("authors", [])
                new_authors = input(f"  Authors [{', '.join(raw_authors)}]: ").strip()
                if new_authors:
                    props["authors"] = [a.strip() for a in new_authors.split(",") if a.strip()]
            elif tbl == "authors":
                new_name = input(f"  Name [{props.get('name', '')}]: ").strip()
                if new_name:
                    props["name"] = new_name
            else:  # concepts
                new_name = input(f"  Name [{props.get('name', '')}]: ").strip()
                if new_name:
                    props["name"] = new_name
            graph_repo.update_node_properties(node_id, props)
            return "[bold green]✓  Record updated[/bold green]"
        except KeyboardInterrupt:
            return "[yellow]Edit cancelled.[/yellow]"

    while True:
        con.console.clear()

        total = _count(active_table)
        total_pages = max(1, math.ceil(total / limit))
        page = max(1, min(page, total_pages))

        rows = _get_rows(active_table, page)

        # ── Build the active table ───────────────────────────────────────────
        tab_title = TABLE_LABELS[active_table]
        border = {"documents": "blue", "authors": "yellow", "concepts": "magenta"}[active_table]
        search_suffix = f" (filtered: '{search_query}')" if search_query else ""
        table = Table(
            title=f"{tab_title}  [dim](page {page}/{total_pages}, {total} total){search_suffix}[/dim]",
            box=box.ROUNDED, border_style=border, expand=True, header_style="bold cyan",
        )
        table.add_column("#", style="dim", max_width=4, justify="right")

        if active_table == "documents":
            table.add_column("Type", style="cyan", max_width=10)
            table.add_column("Title", style="bold white")
            table.add_column("Authors", style="green", max_width=28)
            table.add_column("Path / URL", style="dim")
        elif active_table == "authors":
            table.add_column("Author Name", style="bold yellow")
            table.add_column("Papers", justify="right", style="cyan", max_width=8)
        else:
            table.add_column("Concept", style="bold magenta")
            table.add_column("Mentions", justify="right", style="cyan", max_width=8)

        for i, r in enumerate(rows, start=1):
            num = f"[bold cyan]{i}[/bold cyan]" if selected_idx == i else str(i)
            props = json.loads(r["properties"])

            if active_table == "documents":
                stype  = props.get("source_type", "paper")
                title  = props.get("title", r["id"])
                if len(title) > 60: title = title[:57] + "…"
                authors = ", ".join(props.get("authors", [])) or "—"
                path   = props.get("file_path") or props.get("url") or "—"
                if len(path) > 55: path = "…" + path[-52:]
                row_style = "on grey15" if selected_idx == i else ""
                table.add_row(num, stype, title, authors, path, style=row_style)
            elif active_table == "authors":
                name = props.get("name", r["id"]).replace("_", " ").title()
                cnt  = str(r["papers_count"])
                row_style = "on grey15" if selected_idx == i else ""
                table.add_row(num, name, cnt, style=row_style)
            else:
                raw  = props.get("name", r["id"])
                name = raw.replace("_", " ").title()
                if len(name) > 65: name = name[:62] + "…"
                cnt  = str(r["degree"])
                row_style = "on grey15" if selected_idx == i else ""
                table.add_row(num, name, cnt, style=row_style)

        con.console.print(table)

        # ── Tab bar ──────────────────────────────────────────────────────────
        tab_bar_parts = []
        for t in TABLES:
            label = TABLE_LABELS[t]
            if t == active_table:
                tab_bar_parts.append(f"[bold white on blue] {label} [/bold white on blue]")
            else:
                tab_bar_parts.append(f"[dim] {label} [/dim]")
        con.console.print("  " + "  ".join(tab_bar_parts))
        con.console.print()

        # ── Status / help line ───────────────────────────────────────────────
        if status_msg:
            con.console.print(f"  {status_msg}")
            status_msg = ""
        else:
            if selected_idx is not None:
                row = rows[selected_idx - 1]
                props = json.loads(row["properties"])
                full_path = props.get("file_path") or props.get("url") or "—"
                con.console.print(f"  [bold cyan]Full Path/URL:[/bold cyan] [dim]{full_path}[/dim]")
                
                doc_actions = ""
                if active_table == "documents":
                    doc_actions = "  [bold]A[/bold] Annotation  [bold]O[/bold] Open  [bold]S[/bold] Summary  "
                con.console.print(
                    f"  Row [bold cyan]{selected_idx}[/bold cyan] selected  │{doc_actions}"
                    "[bold]E[/bold] Edit  [bold]X[/bold] Delete  [bold]Esc[/bold] Deselect"
                )
            else:
                con.console.print(
                    "  [bold]←/A[/bold] Prev  [bold]→/D[/bold] Next  "
                    "[bold]Tab[/bold] Switch table  "
                    "[bold]/[/bold] Search  "
                    "[bold]Esc[/bold] Clear filter  "
                    "[bold]1-9…[/bold] Select row  [bold]Q[/bold] Quit"
                )

        # ── Input ────────────────────────────────────────────────────────────
        try:
            c = click.getchar()
        except Exception:
            break

        if c in ('q', 'Q', '\x03', '\x04'):
            break
        elif c in ('a', 'A', '\x1b[D'):           # left / previous page or Annotation if doc selected
            if selected_idx is not None and active_table == "documents":
                row = rows[selected_idx - 1]
                props = json.loads(row["properties"])
                abstract = props.get("abstract") or "No abstract/annotation available."
                title = props.get("title") or row["id"]
                con.console.clear()
                con.console.print(Panel(
                    abstract,
                    title=f"[bold green]Annotation: {title[:60]}[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                ))
                con.console.print("\n  Press any key to return...")
                click.getchar()
            else:
                page = max(1, page - 1)
                selected_idx = None
        elif c in ('d', 'D', '\x1b[C'):           # right / next page
            page = min(total_pages, page + 1)
            selected_idx = None
        elif c == '\t':                            # Tab — cycle tables
            idx = TABLES.index(active_table)
            active_table = TABLES[(idx + 1) % len(TABLES)]
            page = 1
            selected_idx = None
        elif c == '\x1b':                          # Escape — deselect / clear search
            if selected_idx is not None:
                selected_idx = None
            else:
                search_query = None
                page = 1
        elif c == '/':                             # Slash — search query prompt
            con.console.print("\n  [bold yellow]Search query:[/bold yellow] ", end="")
            try:
                query_str = input().strip()
                if query_str:
                    search_query = query_str
                else:
                    search_query = None
                page = 1
                selected_idx = None
            except KeyboardInterrupt:
                pass
        elif c in ('o', 'O') and selected_idx is not None and active_table == "documents":
            row = rows[selected_idx - 1]
            props = json.loads(row["properties"])
            path = props.get("file_path") or props.get("url")
            if not path:
                status_msg = "[red]No file path or URL associated with this document.[/red]"
            else:
                con.console.print(f"\n[green]Opening: {path}[/green]")
                try:
                    if path.startswith("http://") or path.startswith("https://"):
                        webbrowser.open(path)
                        status_msg = f"[green]Opened URL in browser: {path[:50]}[/green]"
                    else:
                        expanded_path = os.path.expanduser(path)
                        if not os.path.exists(expanded_path):
                            expanded_path = str(Path(path).resolve())
                        
                        if sys.platform == "win32":
                            os.startfile(expanded_path)
                        elif sys.platform == "darwin":
                            subprocess.run(["open", expanded_path])
                        else:
                            subprocess.run(["xdg-open", expanded_path])
                        status_msg = f"[green]Opened file locally: {os.path.basename(path)}[/green]"
                except Exception as e:
                    status_msg = f"[red]Failed to open file: {e}[/red]"
        elif c in ('s', 'S') and selected_idx is not None and active_table == "documents":
            row = rows[selected_idx - 1]
            paper_id = row["id"]
            props = json.loads(row["properties"])
            summary = props.get("summary")
            title = props.get("title") or paper_id
            
            if not summary:
                con.console.print("\n[yellow]Generating LLM Summary... This might take a few seconds.[/yellow]")
                try:
                    vector_repo = container.get_vector_repo()
                    llm_engine = container.get_llm_engine(use_cloud=False)
                    
                    chunks = vector_repo.get_chunks_for_paper(paper_id)
                    sample_text = "\n\n".join([ch.text_content for ch in chunks[:5]]) if chunks else ""
                    abstract = props.get("abstract") or ""
                    
                    if not llm_engine:
                        raise ValueError("LLM Engine could not be initialized. Please check your model path/provider config.")
                    
                    prompt = prompts.get_prompt("synthesis", "paper_summary", title=title, abstract=abstract, sample_text=sample_text[:3000])

                    summary = llm_engine.generate_response(prompt)
                    
                    # Save to DB
                    props["summary"] = summary
                    graph_repo.update_node_properties(paper_id, props)
                    con.success("Summary generated and saved to database.")
                except Exception as e:
                    summary = f"Error generating summary: {e}"
            
            con.console.clear()
            con.console.print(Panel(
                summary,
                title=f"[bold cyan]LLM Summary: {title[:60]}[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
                expand=True,
            ))
            con.console.print("\n  Press any key to return...")
            click.getchar()
        elif c.isdigit():                          # digit — start building row number
            val = int(c)
            total_rows = len(rows)
            if val == 0:
                status_msg = "[yellow]Row 0 is invalid.[/yellow]"
            elif val * 10 > total_rows:
                if 1 <= val <= total_rows:
                    selected_idx = val
                else:
                    status_msg = f"[yellow]Row {val} not on this page (1–{total_rows})[/yellow]"
            else:
                con.console.print(f"\n  Row #: [bold cyan]{val}[/bold cyan]  (Enter to select {val}, or press second digit)", end="")
                nc = click.getchar()
                if nc in ('\r', '\n'):
                    selected_idx = val
                elif nc.isdigit():
                    new_val = val * 10 + int(nc)
                    if 1 <= new_val <= total_rows:
                        selected_idx = new_val
                    else:
                        status_msg = f"[yellow]Row {new_val} not on this page (1–{total_rows})[/yellow]"
                else:
                    status_msg = "[yellow]Selection cancelled.[/yellow]"
        elif c in ('e', 'E') and selected_idx is not None:
            row = rows[selected_idx - 1]
            status_msg = _edit_node(row["id"], active_table)
        elif c in ('x', 'X') and selected_idx is not None:
            row = rows[selected_idx - 1]
            props = json.loads(row["properties"])
            name = props.get("title") or props.get("name") or row["id"]
            con.console.print(f"\n  [bold red]Delete[/bold red] [white]{name[:60]}[/white]?  Y/n  ", end="")
            confirm = click.getchar()
            if confirm in ('y', 'Y', '\r', '\n'):
                status_msg = _delete_node(row["id"])
                selected_idx = None
            else:
                status_msg = "[dim]Deletion cancelled.[/dim]"
