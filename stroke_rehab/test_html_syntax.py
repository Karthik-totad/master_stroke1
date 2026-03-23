from html.parser import HTMLParser

class C(HTMLParser): 
    pass

with open('ui/neurorehab_games.html', encoding='utf-8') as f: 
    C().feed(f.read())
    
print('HTML syntax OK')
