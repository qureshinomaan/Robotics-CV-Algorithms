#include<bits/stdc++.h>

typedef long long int ll;

int main(void){
  ll i;
  ll n, e;
  ll start;
  
  cin >> n; // Enter number of nodes.
  cin >> e; // Enter number of edges
  vector <ll> adj[n];
  vector <ll> visited(n, 0);
  vector <ll> dist(n, 0);
  queue <ll> q;
  
  for(i=0; i<e; i++){
    ll node1, node2;
    cin>>node1>>node2;
    adj[node1].push_back(node2);
    adj[node2].push_back(node1);
  }
  cin >> start; // Starting node for BFS.
  
  q.push(start);

  while(!q.empty()){
    int cur_node = q.front();
    
    if(visited[cur_node] != 2){
      
      visited[cur_node] = 2;
      for(ll j = 0; j< adj[cur_node].size(); j++){
        if(visited[adj[cur_node][j]] == 0){
          dist[adj[cur_node][j]] = dist[cur_node] + 1;
          visited[adj[cur_node][j]] = 1;
          q.push(adj[cur_node][j]);
        }
      }
    }
    q.pop();
  }
  for(i=0;i<n;i++)
    cout<<dist[i];
  
  return 0;
}
