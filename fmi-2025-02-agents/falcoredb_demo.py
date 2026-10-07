from falkordb import FalkorDB

# 1. Connect to FalkorDB (Change host/port/password if using FalkorDB Cloud)
db = FalkorDB(host='localhost', port=6379)

# 2. Select or create a test graph
graph = db.select_graph("TestNetwork")

print("--- Testing FalkorDB ---")

# 3. Create nodes and a relationship
print("Creating nodes...")
create_query = """
CREATE (a:Rider {name: 'Rossi', number: 46})
CREATE (b:Team {name: 'Yamaha'})
CREATE (a)-[r:RIDES_FOR]->(b)
RETURN a.name, b.name
"""
result = graph.query(create_query)
print(f"Graph population done.")

# 4. Query the graph data
print("\nQuerying graph...")
match_query = """
MATCH (rider:Rider)-[:RIDES_FOR]->(team:Team)
RETURN rider.name, rider.number, team.name
"""
result = graph.query(match_query)

# 5. Print the formatted output
for record in result.result_set:
    print(f"Rider: {record[0]} (#{record[1]}) rides for Team: {record[2]}")

# 6. Clean up (Optional: Deletes the graph data)
print("\nCleaning up test data...")
graph.delete()
print("Success! FalkorDB is fully functional.")
