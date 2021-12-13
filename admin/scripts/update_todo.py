with open("README.md", encoding="utf-8") as file:
    data = file.read().split("## Features ✔", 1)[1].split("<br>", 1)[0].splitlines()

output = "".join(
    line.replace("\n", "") + "<!-- TODO: " + line.replace("❌", "").replace("\n", "").replace("\t", "").replace("-", "") + " -->\n"
    for line in data if "❌" in line
)

with open("admin/markdown/todo.md", "w", encoding="utf-8") as file:
    file.write(output)