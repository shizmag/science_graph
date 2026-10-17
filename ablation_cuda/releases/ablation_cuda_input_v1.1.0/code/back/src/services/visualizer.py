"""
Visualizer — generates the interactive HTML network graph.
"""

import json
from pathlib import Path
from src.repository.base import GraphRepository


def generate_html_graph(graph_repo: GraphRepository, output_path: Path) -> None:
    """Generates an interactive HTML knowledge graph from repository data."""
    # Get node degrees
    degrees = {}
    edges_rows = graph_repo.get_all_edges()
    for e in edges_rows:
        src_id, tgt_id, etype, _ = e
        degrees[src_id] = degrees.get(src_id, 0) + 1
        degrees[tgt_id] = degrees.get(tgt_id, 0) + 1

    nodes_rows = graph_repo.get_all_nodes()

    if not nodes_rows:
        raise ValueError("Knowledge graph is empty. Index some documents first.")

    # Process nodes
    vis_nodes = []
    for r in nodes_rows:
        node_id, label, props_str = r
        props = json.loads(props_str)
        source_type = props.get("source_type", "paper")
        degree = degrees.get(node_id, 0)

        if label == "Paper":
            title = props.get("title", node_id)
            node_label = title if len(title) < 25 else title[:22] + "..."
            color_map = {"note": "#f03e3e", "book": "#7950f2", "paper": "#4c6ef5", "webpage": "#20c997"}
            color = color_map.get(source_type, "#4c6ef5")
            size = 25
            shape = "dot"
        elif label == "Author":
            node_label = props.get("name", node_id).title()
            color = "#fab005"
            size = 20
            shape = "dot"
        elif label == "Concept":
            raw_name = props.get("name", node_id)
            node_label = raw_name.replace("_", " ").title()
            is_tag = props.get("is_tag", False)
            color = "#da77f2" if is_tag else "#12b886"
            size = 18 if is_tag else 20
            shape = "dot"
        else:
            node_label = node_id
            color = "#868e96"
            size = 15
            shape = "dot"

        vis_nodes.append({
            "id": node_id,
            "label": node_label,
            "title": f"<b>{label}</b>: {props.get('title', props.get('name', node_id))}<br>ID: {node_id}<br>Degree: {degree}",
            "color": color,
            "size": size,
            "shape": shape,
            "group": label,
            "degree": degree,
            "created_at": props.get("created_at"),
            "year": props.get("year"),
        })

    vis_edges = []
    for e in edges_rows:
        src_id, tgt_id, edge_type, _ = e
        color = "#adb5bd"
        dashes = False
        width = 1

        if edge_type == "AUTHORED":
            color = "#ffd43b"
            width = 2
        elif edge_type == "MENTIONS_CONCEPT":
            color = "#69db7c"
            dashes = True
        elif edge_type == "CITES":
            color = "#748ffc"
            width = 2
        elif edge_type == "HAS_TAG":
            color = "#da77f2"
            dashes = True
            width = 1

        vis_edges.append({
            "from": src_id,
            "to": tgt_id,
            "label": edge_type,
            "arrows": "to",
            "font": {"size": 8, "align": "top"},
            "color": {"color": color, "highlight": "#495057"},
            "dashes": dashes,
            "width": width,
        })

    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>Science Graph — Knowledge Network</title>
    <script type="text/javascript">
        // Dynamic loader fallback for vis-network.min.js
        (function() {{
            var urls = [
                "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js",
                "https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js",
                "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"
            ];
            var index = 0;
            function tryLoad() {{
                if (index >= urls.length) {{
                    console.error("Failed to load vis-network from all sources.");
                    return;
                }}
                var script = document.createElement("script");
                script.type = "text/javascript";
                script.src = urls[index];
                script.onload = function() {{
                    console.log("Successfully loaded vis-network from: " + urls[index]);
                    if (window.initGraph) {{
                        window.initGraph();
                    }}
                }};
                script.onerror = function() {{
                    console.warn("Failed to load vis-network from: " + urls[index] + ". Trying next...");
                    index++;
                    tryLoad();
                }};
                document.head.appendChild(script);
            }}
            // Start loading when document is ready
            if (document.readyState === "loading") {{
                document.addEventListener("DOMContentLoaded", tryLoad);
            }} else {{
                tryLoad();
            }}
        }})();
    </script>
    <style type="text/css">
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               margin: 0; background-color: #1a1b1e; color: #c1c2c5; display: flex; flex-direction: column; height: 100vh; }}
        #header {{ padding: 15px 20px; background-color: #25262b; border-bottom: 1px solid #2c2e33; display: flex; justify-content: space-between; align-items: center; }}
        h2 {{ margin: 0; color: #fff; font-size: 1.2rem; }}
        #mynetwork {{ flex: 1; width: 100%; background-color: #1a1b1e; }}
        .legend {{ display: inline-block; margin-right: 15px; font-size: 14px; }}
        .legend-color {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%;
                         margin-right: 5px; vertical-align: middle; }}
        .controls {{ display: flex; align-items: center; gap: 15px; }}
        .slider-container {{ display: flex; align-items: center; gap: 10px; background: #2c2e33; padding: 5px 15px; border-radius: 6px; }}
        input[type=range] {{ cursor: pointer; }}
    </style>
</head>
<body>
    <div id="header">
        <div>
            <h2>🔬 Science Graph — Knowledge Network</h2>
            <div style="margin-top: 8px;">
                <span class="legend"><span class="legend-color" style="background:#4c6ef5"></span>Paper</span>
                <span class="legend"><span class="legend-color" style="background:#f03e3e"></span>Note</span>
                <span class="legend"><span class="legend-color" style="background:#7950f2"></span>Book</span>
                <span class="legend"><span class="legend-color" style="background:#20c997"></span>Webpage</span>
                <span class="legend"><span class="legend-color" style="background:#fab005"></span>Author</span>
                <span class="legend"><span class="legend-color" style="background:#12b886"></span>Concept</span>
                <span class="legend"><span class="legend-color" style="background:#da77f2"></span>Tag</span>
            </div>
        </div>
        <div class="controls">
            <div class="slider-container">
                <label for="yearFilter">Year:</label>
                <select id="yearFilter" onchange="applyFilters()" style="background:#1a1b1e; color:#c1c2c5; border:1px solid #2c2e33; border-radius:4px; padding:3px 8px; outline:none; cursor:pointer;">
                    <option value="all">All Years</option>
                </select>
            </div>
            <div class="slider-container">
                <label for="degreeSlider">Min Connections:</label>
                <input type="range" id="degreeSlider" min="1" max="20" value="3" oninput="applyFilters()">
                <span id="sliderValue" style="font-weight: bold; width: 20px;">3</span>
            </div>
        </div>
    </div>
    <div id="mynetwork"></div>
    <script type="text/javascript">
        var allNodes = {json.dumps(vis_nodes, ensure_ascii=False)};
        var allEdges = {json.dumps(vis_edges, ensure_ascii=False)};
        var nodesView, edgesView, network;

        function initGraph() {{
            nodesView = new vis.DataSet(allNodes);
            edgesView = new vis.DataSet(allEdges);

            network = new vis.Network(
                document.getElementById('mynetwork'),
                {{ nodes: nodesView, edges: edgesView }},
                {{
                    nodes: {{ font: {{ color: '#c1c2c5', size: 12 }} }},
                    edges: {{ smooth: {{ type: 'continuous' }} }},
                    physics: {{ 
                        barnesHut: {{ 
                            gravitationalConstant: -12000,
                            centralGravity: 0.2,
                            springLength: 250,
                            springConstant: 0.04,
                            damping: 0.09,
                            avoidOverlap: 0.3
                        }},
                        stabilization: {{ iterations: 200 }}
                    }}
                }}
            );

            // Populate year options dynamically on init
            var years = new Set();
            allNodes.forEach(n => {{
                if (n.year) {{
                    years.add(n.year);
                }} else if (n.created_at) {{
                    var y = new Date(n.created_at).getFullYear();
                    if (!isNaN(y)) {{
                        years.add(y);
                    }}
                }}
            }});
            var sortedYears = Array.from(years).sort((a,b) => b-a);
            var select = document.getElementById('yearFilter');
            sortedYears.forEach(y => {{
                var opt = document.createElement('option');
                opt.value = y;
                opt.innerText = y;
                select.appendChild(opt);
            }});

            // Initial filter
            var defaultFilter = allNodes.length < 150 ? 1 : 3;
            document.getElementById('degreeSlider').value = defaultFilter;
            applyFilters();
        }}

        function applyFilters() {{
            var val = document.getElementById('degreeSlider').value;
            document.getElementById('sliderValue').innerText = val;
            var minDegree = parseInt(val, 10);
            var yearVal = document.getElementById('yearFilter').value;
            
            // Filter nodes
            var filteredNodes = allNodes.filter(n => {{
                // Check degree
                var degreeMatch = n.degree >= minDegree || n.group === 'Paper';
                if (!degreeMatch) return false;
                
                // Check year
                if (yearVal !== 'all') {{
                    var targetYear = parseInt(yearVal, 10);
                    var nodeYear = n.year;
                    if (!nodeYear && n.created_at) {{
                        var d = new Date(n.created_at);
                        if (!isNaN(d)) nodeYear = d.getFullYear();
                    }}
                    if (nodeYear !== targetYear) return false;
                }}
                return true;
            }});
            
            if (nodesView) {{
                nodesView.clear();
                nodesView.add(filteredNodes);
            }}
            
            var validIds = new Set(filteredNodes.map(n => n.id));
            var filteredEdges = allEdges.filter(e => validIds.has(e.from) && validIds.has(e.to));
            if (edgesView) {{
                edgesView.clear();
                edgesView.add(filteredEdges);
            }}
        }}
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
