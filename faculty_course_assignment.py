import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------
# 1. Load datasets
# ------------------------------------
spring_2025_data = pd.read_excel("Spring 2025 data.xlsx", skiprows=1)
spring_2024_data = pd.read_excel("Spring 2024 data.xlsx", skiprows=1)
spring_2023_data = pd.read_excel("Spring 2023 data.xlsx", skiprows=1)
spring_2022_data = pd.read_excel("Spring 2022 data.xlsx", skiprows=1)
summer_2024_data = pd.read_excel("Summer 2024 data.xlsx", skiprows=1)
summer_2023_data = pd.read_excel("Summer 2023 data.xlsx", skiprows=1)
summer_2022_data = pd.read_excel("Summer 2022 data.xlsx", skiprows=1)
fall_2024_data = pd.read_excel("Fall 2024 data.xlsx", skiprows=1)
fall_2023_data = pd.read_excel("Fall 2023 data.xlsx", skiprows=1)
fall_2022_data = pd.read_excel("Fall 2022 data.xlsx", skiprows=1)
fall_2021_data = pd.read_excel("Fall 2021 data.xlsx", skiprows=1)
ug_2023_fall = pd.read_excel("Undergraduate Fall 2023 data.xlsx", skiprows=1)
ug_2023_summer = pd.read_excel("Undergraduate Summer 2023 data.xlsx", skiprows=1)
ug_2023_spring = pd.read_excel("Undergraduate 2023 spring.xlsx", skiprows=1)
ug_2024_spring = pd.read_excel("Undergraduate Spring 2024 data.xlsx", skiprows=1)
ug_2024_summer = pd.read_excel("Undergraduate Summer 2024 data.xlsx", skiprows=1)
ug_2024_fall = pd.read_excel("Undergraduate Fall 2024.xlsx", skiprows=1)

# ------------------------------------
# 2. Rename columns, clean, and deduplicate
# ------------------------------------
columns = ["Course Code", "Course", "Faculty Name"]

# Define cleaning function
def clean_text(val):
    if isinstance(val, str):
        return val.replace(u'\xa0', u' ').strip()
    return val

datasets = [
    fall_2024_data, spring_2025_data, summer_2024_data,
    spring_2024_data, spring_2023_data, spring_2022_data,
    summer_2023_data, summer_2022_data,
    fall_2023_data, fall_2022_data, fall_2021_data,
    ug_2023_fall, ug_2023_summer, ug_2023_spring,
    ug_2024_spring, ug_2024_summer, ug_2024_fall
]

for i, df in enumerate(datasets):
    df.columns = columns

    # Clean text columns
    df["Faculty Name"] = df["Faculty Name"].apply(clean_text)
    df["Course"] = df["Course"].apply(clean_text)

    # Remove duplicates after cleaning
    datasets[i] = df.drop_duplicates(subset=columns)

# ------------------------------------
# 3. Combine datasets
# ------------------------------------
combined_data = pd.concat(datasets, ignore_index=True)


# ------------------------------------
# 4. Filter out non-faculty labels
# ------------------------------------
non_faculty_labels = {
    "Not available",
    "Course not offered in spring",
    "Course not offered in Spring",
    "Course not offered in Summer",
    "Course not offered in Fall"
}

filtered_data = combined_data[~combined_data["Faculty Name"].isin(non_faculty_labels)]

# ------------------------------------
# 6. Load Spring 2025 Faculty and Class Sections
# ------------------------------------
faculty_2025 = pd.read_excel("2025 Faculty.xlsx")
classes_2025 = pd.read_excel("2025 class and sections.xlsx")

# ------------------------------------
# 7. Expand course sections
# ------------------------------------
course_sections = []
for _, row in classes_2025.iterrows():
    num_sections = int(row['Number_of_Sections'])
    for section_num in range(num_sections):
        course_sections.append({
            'Course_Name': row['Course_Name'],
            'Section_ID': f"{row['Course_Name']}_Section_{section_num + 1}"
        })

courses_df = pd.DataFrame(course_sections)

# Apply cleaning
courses_df['Course_Name'] = courses_df['Course_Name'].apply(clean_text)

# ------------------------------------
# 8. Matching rounds
# ------------------------------------
assignments = []
remaining_courses = courses_df.copy()

for round_num in range(3):  # Run 3 rounds of matching
    if remaining_courses.empty:
        break

    num_faculty = len(faculty_2025)
    num_courses = len(remaining_courses)
    adjacency_matrix = np.zeros((num_faculty, num_courses))

    for i, faculty in faculty_2025.iterrows():
        faculty_name = clean_text(faculty['Faculty_Name'])

        for j, course in remaining_courses.iterrows():
            course_name = clean_text(course['Course_Name'])

            experience_count = len(filtered_data[
                (filtered_data['Faculty Name'] == faculty_name) &
                (filtered_data['Course'] == course_name)
            ])

            adjacency_matrix[i][j] = -experience_count

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(adjacency_matrix)

    round_assignments = []
    for faculty_idx, course_idx in zip(row_ind, col_ind):
        if faculty_idx < num_faculty and course_idx < num_courses:  # Avoid index errors
            faculty_name = faculty_2025.iloc[faculty_idx]['Faculty_Name']
            course_section = remaining_courses.iloc[course_idx]

            round_assignments.append({
                'Round': round_num + 1,
                'Faculty_Name': faculty_name,
                'Course_Name': course_section['Course_Name'],
                'Section_ID': course_section['Section_ID'],
                'Experience_Score': -adjacency_matrix[faculty_idx, course_idx]
            })

    assignments.extend(round_assignments)

    # Remove matched sections
    matched_sections = {assign['Section_ID'] for assign in round_assignments}
    remaining_courses = remaining_courses[~remaining_courses['Section_ID'].isin(matched_sections)].reset_index(drop=True)

# Convert assignments to DataFrame
assignments_df = pd.DataFrame(assignments)

# ------------------------------------
# 9. Export assignments
# ------------------------------------
assignments_df.to_excel("Spring_2025_Faculty_Assignments_Rounded.xlsx", index=False)
print("Assignments successfully saved to Spring_2025_Faculty_Assignments_Rounded.xlsx!")

# ------------------------------------
# 10. Visualization: Faculty ↔ Course with Completely Different Adjacent Edge Colors
# ------------------------------------
import matplotlib.colors as mcolors
import numpy as np

# Define faculty and course nodes based on the assignments DataFrame
faculty_nodes = assignments_df["Faculty_Name"].unique()
course_nodes = assignments_df["Course_Name"].unique()

# Define edges directly between Faculty and Course
edges = list(zip(assignments_df["Faculty_Name"], assignments_df["Course_Name"]))

# Create the bipartite graph
B = nx.Graph()
B.add_nodes_from(faculty_nodes, bipartite=0)  # Faculty layer
B.add_nodes_from(course_nodes, bipartite=1)   # Courses layer
B.add_edges_from(edges)

# Assign a unique color to each adjacent edge for each node
edge_colors = []
for node in B.nodes:
    adjacent_edges = list(B.edges(node))  # Get edges connected to this node
    unique_colors = list(mcolors.TABLEAU_COLORS)  # List of predefined distinct colors

    for idx, edge in enumerate(adjacent_edges):
        if edge[0] == node:
            edge_colors.append(unique_colors[idx % len(unique_colors)])  # Wrap colors if needed
        elif edge[1] == node:
            edge_colors.append(unique_colors[idx % len(unique_colors)])

# Generate a bipartite layout
pos = nx.bipartite_layout(B, faculty_nodes)
scale_factor = 1.5
for node, coord in pos.items():
    pos[node] = (coord[0] * 2, coord[1] * scale_factor)

# Plot the graph
plt.figure(figsize=(24, 16))
nx.draw_networkx_nodes(B, pos, nodelist=faculty_nodes, node_color="skyblue", node_size=1800, label="Faculty")
nx.draw_networkx_nodes(B, pos, nodelist=course_nodes, node_color="lightgreen", node_size=2000, label="Courses")
nx.draw_networkx_edges(B, pos, edgelist=edges, edge_color=edge_colors, alpha=0.6, width=2)  # Different edge colors
nx.draw_networkx_labels(B, pos, font_size=12)

# Add title and legend
plt.legend(loc="upper left", fontsize=14)
plt.title("Faculty-Course Bipartite Graph with Completely Different Adjacent Edges Colored", fontsize=22)
plt.axis("off")
plt.savefig("faculty_course_bipartite_graph_unique_edges.png", dpi=300, bbox_inches="tight")
plt.show()
