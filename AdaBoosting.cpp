#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>

using namespace std;

double wt[32561];
const int NUM_FIN=20;
double wt_tre[NUM_FIN];
int NUMATTR = 14;
int DATASIZE = 32561;
int TDATASIZE = 16281;
const int NUMTREES = 50;
int picked[NUMTREES]={0};
int id3_fin[NUM_FIN];
int RDATASIZE = 1000;
int RNUMATTR = 5;

// This is the global poiner to all data
int (*dat)[15];
// This will store max value of attributes starting from 0
int ranges[15]={1,7,1,15,1,6,13,5,4,1,1,1,1,40,1};

const int numattr = NUMATTR;

// This function is used in sorting 2d array, a and b are indexes of data
// sc variable above is on basis of which column 2d array is sorted
int sc;

bool comp(int a,int b)
{
	return dat[a][sc]<dat[b][sc];
}

// Node object represents creating nodes in decision tree
// attr represents atrribute that node is holding
// value refers to true or false this is valid only for leaf node where attr = -1
// element at index i in childs vector will point a decision tree if value of attr is i 
class Node
{
public:
	int attr;
	bool value;
	vector<Node*> childs;
	Node(int a,bool v,int r)
	{
		attr = a; 
		value = v;
		childs.insert(childs.begin(),r,0);
	}
};

// This object represents :
// training data in form of data vector, which will store index of all instances to consider for constructing tree
// attributes in form of vis vector,
// final tree whose root is stored in root
class ID3
{
private:
	vector<int> data;
	vector<bool> vis;
	int dataSize, mismatches, numattr,tdataSize;
	Node *root;

	// this function calculates entropy for data from s to e 
	// in data vector and for column c
	double entLoss(int s, int e,int c)
	{
		int j,numvalues,k;
		double ent,prob,loss;

		numvalues = ranges[c]+1;
		int t[numvalues] ={}, f[numvalues]={};
		for(j=s;j<=e;j++)
		{
			k = data[j];
			t[dat[k][c]] += dat[k][numattr];
			f[dat[k][c]] += !dat[k][numattr];
		}

		for(loss=0,j=0;j<numvalues;j++)
		{
			if(t[j]!=0 && f[j] !=0)
			{
				prob = (t[j]*1.0)/(t[j]+f[j]);
				ent = -(prob*log(prob)+(1-prob)*log(1-prob));
				loss += (ent*((t[j]+f[j]))/(e-s+1));
			}
		}

		return loss;
	}

	// this function will find decision tree for data from s to e and 
	// attributes which has false in vis vector and return node of that tree
	// This is done by first finding which attrubute has low entropy and sorting 
	// based on that column and dividing data into segments which has same 
	// value for that attribute and recursively finding root of that segment leaving
	// already visited attrubites
	Node* train(int s, int e)
	{	
		// printf("aaaaaaa\n");
		int minc, i, j, k, p, c, cnt, numvalues,ju;
		double min, loss;
		
		// printf("processing %d %d\n",s,e);
		min = 9999;

		for(i=s, cnt=0; i<=e; i++)
		{
			k = data[i];
			// printf("%d %d\n",k,numattr);
			// fflush(stdout);
			cnt = cnt+dat[k][numattr];
		}

		if(cnt==0)
		{
			// printf("false");
			return new Node(-1,false,0);
		}
		else if(cnt==e-s+1)
		{
			// printf("true");
			return new Node(-1,true,0);
		}

		// find minimum entropy column number and store
		// in  minc
		for(i=0,j=0;i<numattr;i++)
		{
			if(vis[i])
				continue;
			j=1;
			loss = entLoss(s,e,i);
			if(loss<min)
			{
				min = loss;
				minc = i;
			}
		}

		if(j==0)
		{
			// printf("j==0\n");
			if(cnt>((e-s+1)/2))
				return new Node(-1,true,0);
			else
				return new Node(-1,false,0);
		}

		Node &n = *(new Node(minc,true,ranges[minc]+1));
		
		sc = minc;
		sort(data.begin()+s,data.begin()+e+1,comp);
		
		// printf("best classifier %d\n",minc);
		// print_data();
		vis[minc]  = true;
		numvalues = ranges[minc]+1;
		k = s;
		p = dat[data[s]][minc];

		for(j=s+1;j<=e;j++)
		{
			i = data[j];
			c = dat[i][minc];
			if(p != c)
			{
				// cout<<"going to process"<<k<<" "<<j-1<<"in column"<<minc<<endl;
				// cout.flush();
				// cin>>ju;
				n.childs[p]=train(k,j-1);
				k = j;
			}
			p = c;
		}

		// cout<<"going to process"<<k<<" "<<j-1<<"in column"<<minc<<endl;
		// cout.flush();
		// cin>>ju;
		n.childs[p]=train(k,j-1);

		for(j=0;j<numvalues;j++)
		{
			if(n.childs[j]==0)
			{
				if(cnt>((e-s+1)/2))
					n.childs[j] = new Node(-1,true,0);
				else
					n.childs[j] = new Node(-1,false,0);
			}
		}
		vis[minc] = false;

		return &n;
	}

	// this function will print tree that is genertated
	void print_tree(Node *c)
	{
		int i=0;
		Node *n;
		// printf("abc\n");
		for(Node*n:c->childs)
		{
			if(n->attr ==-1)
			{
				if((n->value)==true)
					printf("attribute-%d value-%d child-true\n",c->attr,i);
				else
					printf("attribute-%d value-%d child-false\n",c->attr,i);
			}
			else
			{
				printf("attribute-%d value-%d child-%d\n",c->attr,i,n->attr);
				print_tree(n);
			}
			i++;
		}
	}

public:
	// this constructor will take total datasize, total number of attributes
	// size of testing data
	ID3(int d,int a,int t,int rd,int rn)
	{
		numattr = a;
		tdataSize = t;
	
		srand(clock());
		set<int> st;

		while(st.size()<rd)
		{
			st.insert(rand()%d);
		}
		for(int i: st)
			data.push_back(i);
		dataSize = data.size();

		srand(clock());
		st.clear();
		for(int i=0;i<a;i++)
			vis.push_back(true);
		while(st.size()<rn)
		{
			st.insert(rand()%a);		
		}
		for(int i: st)
			vis[i] = false;
	}

	ID3(int d,int a,int t)
	{
		int i;
		for(i=0;i<d;i++)
			data.push_back(i);
		for(i=0;i<a;i++)
			vis.push_back(false);
		dataSize = d;
		numattr = a;
		tdataSize = t;
	}

	void print_data()
	{
		int i,j,k;

		// cout<<"ranges"<<endl;
		// for(i=0;i<=numattr;i++)
		// 	cout<<ranges[i]<<" ";
		// cout<<endl;

		printf("Input - Ouptut\n");
		for(i=0;i<dataSize;i++)
		{
			k = data[i];
			for(j=0;j<=numattr;j++)
				printf("%d ",dat[k][j]);
			printf("\n");
		}
	}

	// This function will train tree by calling train(s,e)
	// before calling this function data has to be loaded into
	// "dat" 2D array
	void train()
	{
		root = train(0,dataSize-1);
	}

	void print_tree()
	{
		print_tree(root);
	}

	// This function will evaluate i th data instance on this tree
	bool evaluate(int q)
	{
		Node *n = root;
		while(n->attr!=-1)
			n = n->childs[dat[q][n->attr]];
		return n->value;
	}
	
	// calculate weighted error over all instances
	double treErr()
	{
		int i, k;
		double cnt=0.0;
		mismatches = 0;
		for(i=0;i<DATASIZE;i++)
		{
			if(dat[i][numattr] != evaluate(i))
				cnt+=wt[i];
		}
		return cnt;
	}

	// updates weight of each instance
	void update(double alpha)
	{
		int i, k;
		for(i=0;i<DATASIZE;i++)
		{	
			if(dat[i][numattr] != evaluate(i))
			{ 	// if it misclassifies we will increase weight of instance
				wt[i]=wt[i]*exp(alpha);
			}
			else
			{  //if it correctly classifies we will decrease weight of instance
				wt[i]=wt[i]*exp(-alpha);
			}
		}
	}

	// This function will validate on testing data before calling this function
	// testing data has to be loaded into "dat" 2D array
	void validate()
	{
		int i, cnt=0;
		mismatches = 0;
		for(i=0;i<tdataSize;i++)
		{
			// k = data[i];
			if(dat[i][numattr] != evaluate(i))
				cnt++;
		}
		mismatches = cnt;
		printf("Mismatched %d, Test Data size %d, Attributes %d, Accuracy %lf%%\n",mismatches,tdataSize,numattr, (1-((double)mismatches/tdataSize))*100);
	}

};
// make sum of weights of trees equal to 1
void norm()
{
	int i;
	double sum = 0;
	for(i=0;i<NUM_FIN;i++)
	{
		sum+=wt_tre[i];
	}
	for(i=0;i<NUM_FIN;i++)
	{
		wt_tre[i]/=sum;
	}
}

// This function will load data from file into "dat" 2D array
int loadAllData(int src)
{
	int i,j;
	FILE *f ;
	if(src==1)
		f = fopen("adultdisc.data","r");
	else if(src==2)
		f = fopen("adultdisc.test","r");
	else
		return 0;
	for(i=0;i<DATASIZE;i++)
	{
		for(j=0;j<15;j++)
		{
			fscanf(f,"%d",&dat[i][j]);
		}
	}
	fclose(f);
}
int main()
{
	bool o,t;
	int mismatches,i,j,k,best_k;
	double sum,cur_min,min,cnt;
	dat = (int (*)[15]) malloc(sizeof(int)*(15)*DATASIZE);
	loadAllData(1);
	
	clock_t start = clock();

	// create ID3 pointers array
	ID3 *id3[NUMTREES];

	for(i=0;i<NUMTREES;i++)
	{
		id3[i] = new ID3(DATASIZE,NUMATTR,TDATASIZE,RDATASIZE,RNUMATTR);
		id3[i]->train();
		// id3[i]->selfValidate();
	}
	//adaboosting 
	for(k=0;k<DATASIZE;k++)
	{
		wt[k]=1.0/DATASIZE; // weights of instance initialized to 1/N ,N=no of instances
	}
	for(k=0;k<NUM_FIN;k++) // pick a set of classifiers from the random forest
	{
		min=DATASIZE;
		cur_min=0.0;
		sum=0.0;
		for(i=0;i<DATASIZE;i++)
		{
			sum+=wt[i]; //sum of weights of all instances
		}
		for(i=0;i<NUMTREES;i++) // pick tree that minimizes sum of weighted missclassification
		{
			if(picked[i]==0)
			{
				cur_min=id3[i]->treErr();
				if(cur_min<min)
				{
					best_k=i;
					min=cur_min;
				}
			}
		}
		picked[best_k]=1;
		id3_fin[k]=best_k;
		wt_tre[k]=0.5*log((sum-min)/min); // weight of picked tree
		id3[best_k]->update(wt_tre[k]); // update instace weights
	}
	norm();
	clock_t end = clock();
	printf("Learning time - %lf seconds\n",((double)(end - start))/CLOCKS_PER_SEC);

	// load testing data and test on that data
	loadAllData(2);
	mismatches = 0;
	for(i=0;i<TDATASIZE;i++)
	{
		for(j=0, cnt=0;j<NUM_FIN;j++)
		{
			k=id3_fin[j];
			if(id3[k]->evaluate(i))
			{
				cnt+=wt_tre[j];
			}
			else
			{
				cnt-=wt_tre[j];
			}
		}
		
		o = (cnt>0)?true:false;
		t = dat[i][NUMATTR]?true:false;
		if(o^t)
		{
			mismatches++;
		}
	}
	double err = ((double)mismatches)/TDATASIZE;
	printf("Trees %d, Attributes %d, Datasize %d, Accuracy %lf%%\n",NUMTREES,RNUMATTR,RDATASIZE,(1-err)*100);
}