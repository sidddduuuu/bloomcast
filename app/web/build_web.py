"""Rebuild self-contained index.html from template + network data."""
import pathlib
here=pathlib.Path(__file__).parent
tpl=(here/"template.html").read_text()
data=(here.parent/"bloomcast_network.json").read_text()
(here/"index.html").write_text(tpl.replace("/*__DATA__*/",data))
print("rebuilt index.html")
