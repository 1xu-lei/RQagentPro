import json, os

graph_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'graph_export.json')
with open(graph_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

nodes = data['nodes']
edges = data['edges']

# Collect Domain nodes (use Chinese names as labels)
domain_nodes = [n for n in nodes if n['type'] == 'Domain']

# Collect SubLabel nodes - build "域名-子标签名" as label
sublabel_nodes = [n for n in nodes if n['type'] == 'SubLabel']

# Map sublabel to its domain
sub_to_domain_code = {}
sub_to_domain_name = {}
for n in sublabel_nodes:
    code = n['id'].replace('SubLabel:', '')
    domain_code = code.split('-')[0] if '-' in code else ''
    sub_to_domain_code[n['id']] = domain_code

# Build Domain code -> Domain name mapping from has_sub edges
domain_name_by_id = {n['id']: n.get('label', n['id']) for n in domain_nodes}
domain_code_to_id = {}
for d in domain_nodes:
    name = d.get('label', d['id'])
    # Find which domain code this maps to via taxonomy
    domain_code_to_id[name] = d['id']

# Actually, let's use the taxonomy.py mapping to get domain code -> name
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.domain.taxonomy import DOMAINS, NAME_TO_CODE
code_to_domain_name = {v: k for k, v in NAME_TO_CODE.items()}

# Map sublabel code -> domain name
sublabel_label_map = {}
for n in sublabel_nodes:
    code = n['id'].replace('SubLabel:', '')
    domain_code = code.split('-')[0] if '-' in code else ''
    domain_name = code_to_domain_name.get(domain_code, domain_code)
    sub_name = n.get('label', code)
    # Make the label "域名-子标签名"
    sublabel_label_map[n['id']] = f"{domain_name}-{sub_name}"

# Collect Regulation nodes
regulation_nodes = [n for n in nodes if n['type'] == 'Regulation']

# Build regulation name mapping from taxonomy
from src.domain.taxonomy import REGULATION_MAP  # noqa: E402
reg_label_map = {}
for n in regulation_nodes:
    fg_code = n['id'].replace('Regulation:', '')
    reg_info = REGULATION_MAP.get(fg_code)
    if reg_info:
        reg_label_map[n['id']] = reg_info['title'].replace('《城镇燃气管理条例》', '')
    else:
        reg_label_map[n['id']] = n.get('label', fg_code)

# Domain -> SubLabel edges
d2s_edges = [e for e in edges if e['relation'] == 'has_sub']

# SubLabel -> Regulation (via QA chain, deduplicated)
sub_to_q = {}
for e in edges:
    if e['relation'] == 'has_qa' and e['source'].startswith('SubLabel:'):
        sub_to_q.setdefault(e['source'], set()).add(e['target'])

q_to_a = {e['source']: e['target'] for e in edges
          if e['relation'] == 'has_answer' and e['source'].startswith('Q:')}
a_to_r = {e['source']: e['target'] for e in edges
          if e['relation'] == 'cites' and e['source'].startswith('A:')}

sub_to_regs = {}
for sub, qs in sub_to_q.items():
    regs = set()
    for q in qs:
        a = q_to_a.get(q)
        if a and a in a_to_r:
            regs.add(a_to_r[a])
    if regs:
        sub_to_regs[sub] = regs

# Count edges between each pair to show edge weight
s2r_count = {}
for sub, regs in sub_to_regs.items():
    for r in regs:
        s2r_count[(sub, r)] = len([q for q in sub_to_q.get(sub, set())
                                   if q_to_a.get(q) and a_to_r.get(q_to_a[q]) == r])

# Build graph nodes
graph_nodes = []
graph_edges = []

for n in domain_nodes:
    graph_nodes.append({
        'id': n['id'],
        'label': n.get('label', n['id']),
        'color': '#4CAF50',
        'shape': 'square',
        'size': 20,
        'title': '',
    })

for n in sublabel_nodes:
    graph_nodes.append({
        'id': n['id'],
        'label': sublabel_label_map.get(n['id'], n.get('label', '')),
        'color': '#F44336' if n.get('safety') else '#2196F3',
        'shape': 'dot',
        'size': 14 if n.get('safety') else 12,
        'title': '',
    })

for n in regulation_nodes:
    graph_nodes.append({
        'id': n['id'],
        'label': reg_label_map.get(n['id'], n['id']),
        'color': '#FF9800',
        'shape': 'dot',
        'size': 12,
        'title': '',
    })

# Domain -> SubLabel edges
for e in d2s_edges:
    graph_edges.append({
        'from': e['source'],
        'to': e['target'],
        'arrows': 'to',
        'width': 1,
        'title': '1条',
    })

# SubLabel -> Regulation edges
for (sub, reg), count in s2r_count.items():
    graph_edges.append({
        'from': sub,
        'to': reg,
        'arrows': 'to',
        'width': min(3, max(1, count // 3)),
        'title': f'{count}条',
    })

# Generate HTML matching reference style
html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/css/bootstrap.min.css" rel="stylesheet" crossorigin="anonymous" />
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/js/bootstrap.bundle.min.js" crossorigin="anonymous"></script>

    <center><h1></h1></center>
    <style type="text/css">
        #mynetwork {
            width: 100%;
            height: 700px;
            background-color: #ffffff;
            border: 1px solid lightgray;
            position: relative;
            float: left;
        }
    </style>
</head>

<body>
    <div class="card" style="width: 100%">
        <div id="mynetwork" class="card-body"></div>
    </div>

    <script type="text/javascript">
        var edges;
        var nodes;
        var allNodes;
        var allEdges;
        var nodeColors;
        var originalNodes;
        var network;
        var container;
        var options, data;

        function drawGraph() {
            var container = document.getElementById('mynetwork');

            nodes = new vis.DataSet(''' + json.dumps(graph_nodes, ensure_ascii=False) + r''');
            edges = new vis.DataSet(''' + json.dumps(graph_edges, ensure_ascii=False) + r''');

            nodeColors = {};
            allNodes = nodes.get({ returnType: "Object" });
            for (nodeId in allNodes) {
                nodeColors[nodeId] = allNodes[nodeId].color;
            }
            allEdges = edges.get({ returnType: "Object" });

            data = {nodes: nodes, edges: edges};

            var options = {
                "nodes": {
                    "font": {"size": 14, "face": "MicrosoftYaHei,PingFangSC,sans-serif"}
                },
                "edges": {
                    "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
                },
                "physics": {
                    "enabled": true,
                    "barnesHut": {
                        "gravitationalConstant": -3000,
                        "centralGravity": 0.3,
                        "springLength": 150
                    },
                    "stabilization": {"iterations": 200}
                },
                "interaction": {
                    "navigationButtons": true,
                    "keyboard": true
                }
            };

            network = new vis.Network(container, data, options);
            return network;
        }
        drawGraph();
    </script>
</body>
</html>
'''

output_path = os.path.join(os.path.dirname(__file__), 'graph_high_level.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

safety_count = sum(1 for n in graph_nodes if n.get('color') == '#F44336' and n.get('shape') == 'dot' and n.get('size', 12) >= 14)
print(f"Done: {output_path}")
print(f"Nodes: {len(graph_nodes)} (Domains:{len(domain_nodes)}, SubLabels:{len(sublabel_nodes)}, Regulations:{len(regulation_nodes)})")
print(f"Edges: {len(graph_edges)} (D→S:{len(d2s_edges)}, S→R:{len(s2r_count)})")
