from pylatex import Document, HRule, Section

doc = Document()
Section("First section").insert(doc)

HRule().set_color("green")

print(doc)
