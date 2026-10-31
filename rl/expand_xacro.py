#!/usr/bin/env python3
"""
expand_xacro.py - tiny stand-alone xacro expander (only <xacro:arg>, <xacro:property>, $(arg x) and ${expr}).
Lets you produce a plain .urdf from diffdrive_bot.xacro without ROS installed:
    python3 expand_xacro.py diffdrive_bot.xacro diffdrive_bot.urdf [name:=value ...]
Inside ROS you would normally use:  xacro diffdrive_bot.xacro > diffdrive_bot.urdf
"""
import math
import re
import sys
import xml.etree.ElementTree as ET

XNS = '{http://www.ros.org/wiki/xacro}'


def fmt(v):
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(int(v)) if float(v).is_integer() else format(v, '.6g')
    return str(v)


def expand(path, overrides=None):
    root = ET.parse(path).getroot()
    args, props = {}, {'pi': math.pi}
    ns = {k: getattr(math, k) for k in ('sin', 'cos', 'tan', 'sqrt', 'atan2', 'radians', 'degrees')}

    def evaluate(expr):
        return eval(expr, {'__builtins__': {}}, {**ns, **props})

    def subst(text):
        text = re.sub(r'\$\(arg ([^)]+)\)', lambda m: fmt(args[m.group(1).strip()]), text)
        def repl(m):
            return fmt(evaluate(m.group(1)))
        return re.sub(r'\$\{([^}]*)\}', repl, text)

    def to_val(s):
        try:
            return float(s)
        except ValueError:
            return s

    def truth(s):
        s = s.strip().lower()
        return s in ('true', '1') or (s not in ('false', '0', '') and float(s) != 0.0)

    for el in list(root):
        if el.tag == XNS + 'arg':
            n = el.get('name')
            args[n] = (overrides or {}).get(n, to_val(el.get('default')))
        elif el.tag == XNS + 'property':
            props[el.get('name')] = to_val(subst(el.get('value')))

    def resolve_if(el):
        for ch in list(el):
            if ch.tag in (XNS + 'if', XNS + 'unless'):
                cond = truth(subst(ch.get('value')))
                cond = cond if ch.tag == XNS + 'if' else not cond
                idx = list(el).index(ch)
                el.remove(ch)
                if cond:
                    for j, g in enumerate(list(ch)):
                        el.insert(idx + j, g)
            resolve_if(ch)

    resolve_if(root)

    def walk(el):
        for k, v in list(el.attrib.items()):
            el.set(k, subst(v))
        if el.text and '$' in el.text:
            el.text = subst(el.text)
        for ch in list(el):
            if ch.tag.startswith(XNS):
                el.remove(ch)
            else:
                walk(ch)
    walk(root)
    return root


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    ov = {}
    for a in sys.argv[3:]:
        k, v = a.split(':=')
        try:
            ov[k] = float(v)
        except ValueError:
            ov[k] = v
    root = expand(src, ov)
    ET.indent(root)
    ET.ElementTree(root).write(dst, encoding='unicode', xml_declaration=True)
    print('wrote', dst)
