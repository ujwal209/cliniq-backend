import re

with open('app/routes/chat.py', 'rb') as f:
    content = f.read().decode('utf-8-sig')

old_block = r'    cursor = db\.chat_sessions\.find\(query.*?return history_list'
new_block = (
    '    cursor = db.chat_sessions.find(\n'
    '        query,\n'
    '        {"_id": 0, "session_id": 1, "patient_email": 1, "updated_at": 1,\n'
    '         "is_archived": 1, "custom_title": 1, "messages": {"$slice": 3}}\n'
    '    ).sort("updated_at", -1)\n'
    '    sessions = await cursor.to_list(length=100)\n'
    '\n'
    '    history_list = []\n'
    '    for s in sessions:\n'
    '        title = s.get("custom_title")\n'
    '        if not title:\n'
    '            for m in s.get("messages", []):\n'
    '                if m.get("role") == "user":\n'
    '                    raw = m.get("content", "")\n'
    '                    title = (raw[:50] + "\u2026") if len(raw) > 50 else raw\n'
    '                    break\n'
    '        if not title:\n'
    '            title = "New Chat"\n'
    '        history_list.append({\n'
    '            "id": s["session_id"],\n'
    '            "title": title,\n'
    '            "date": s.get("updated_at", datetime.utcnow()).timestamp() * 1000,\n'
    '            "is_archived": s.get("is_archived", False),\n'
    '            "has_custom_title": bool(s.get("custom_title"))\n'
    '        })\n'
    '    return history_list'
)

new_content = re.sub(old_block, new_block, content, flags=re.DOTALL)
if new_content == content:
    print('NO CHANGE - pattern not matched')
else:
    with open('app/routes/chat.py', 'wb') as f:
        f.write(new_content.encode('utf-8'))
    print('OK - file updated')
