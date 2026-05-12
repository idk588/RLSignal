import xml.etree.ElementTree as ET

tree = ET.parse('Malta/data/malta.net.xml')
root = tree.getroot()

lines = ['<additional>']
for edge in root.findall('edge'):
    edge_id = edge.get('id')
    if edge_id is None or edge_id.startswith(':'):
        continue
    for lane in edge.findall('lane'):
        lane_id = lane.get('id')
        length = float(lane.get('length', 100))
        det_length = min(length, 100)
        lines.append(f'    <laneAreaDetector id="{lane_id}" lane="{lane_id}" pos="0" length="{det_length:.2f}" freq="1" file="ild_out.xml"/>')
lines.append('</additional>')

with open('Malta/data/exp.add.xml', 'w') as f:
    f.write('\n'.join(lines))
print('Done! Written', len(lines)-2, 'detectors')