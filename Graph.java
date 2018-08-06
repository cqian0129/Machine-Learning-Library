import java.io.*;
import java.util.*;

//Program performs graph based clustering class
public class Graph
{
	/************************************************************************/

	//Record class
	private class Record
	{
		//attributes of record
		private double[] attributes;

		//constructor of record
		private Record(double[] attributes)
		{
			this.attributes = attributes;
		}
	}	

	/************************************************************************/

	private int numberRecords; 				//number of records
	private int numberAttributes;			//number of attributes
	private int numberClusters;				//number of clusters
	private double delta;					//neighbor threshold

	private ArrayList<Record> records;		//list of records
	private int[][] matrix;					//adjacency matrix
	private int[] clusters;					//clusters of records

	/************************************************************************/

	//Constructor of clustering
	public Graph() 
	{
		//parameters are zero
		numberRecords = 0;
		numberAttributes = 0;
		delta = 0;

		//lists are empty
		records = null;
		matrix = null;
		clusters = null;
	}

	/************************************************************************/

	//Method loads records from input file
	public void load(String inputFile) throws IOException
	{
		Scanner inFile = new Scanner(new File(inputFile));

		//read number of records, attributes
		numberRecords = inFile.nextInt();
		numberAttributes = inFile.nextInt();

		//empty list of records
		records = new ArrayList<Record>();

		//for each record
		for (int i = 0; i < numberRecords; i++)
		{
			//read attributes
			double[] attributes = new double[numberAttributes];
			for (int j = 0; j < numberAttributes; j++)
				attributes[j] = inFile.nextDouble();

			//create record
			Record record = new Record(attributes);

			//add record to list
			records.add(record);
		}

		inFile.close();
	}

	/************************************************************************/

	//Method sets parameter of clustering
	public void setParameter(double delta)
	{
		//set neighbor threshold
		this.delta = delta;
	}

	/************************************************************************/

	//Method performs clustering
	public void cluster()
	{
		//create adjacency matrix of records
		createMatrix();

		//initialize clusters of records
		initializeClusters();

		//initial record index is 0
		int index = 0;

		//initial cluster name is 0
		int clusterName = 0;

		//while there are more records
		while(index < numberRecords)
		{
			//if record does not have cluster name
			if (clusters[index] == -1)
			{
				//assign cluster name to record and all records connected to it
				assignCluster(index, clusterName);

				//find next cluster name
				clusterName = clusterName + 1;
			}

			//go to next record
			index = index + 1;
		}
		
		numberClusters = clusterName;
	}

	/************************************************************************/

	//Method creates adjacency matrix
	private void createMatrix()
	{
		//allocate adjacency matrix
		matrix = new int[numberRecords][numberRecords];

		//entry (i, j) is 0 or 1 depending on i and j are neighbors or not
		for (int i = 0; i < numberRecords; i++)
			for (int j = 0; j < numberRecords; j++)
				matrix[i][j] = neighbor(records.get(i), records.get(j));
	}

	/************************************************************************/

	//Method decides whether two records are neighbors or not
	private int neighbor(Record u, Record v)
	{
		double distance = 0;

		//find euclidean distance between two records
		for (int i = 0; i < u.attributes.length; i++)
			distance += (u.attributes[i] - v.attributes[i])*
						(u.attributes[i] - v.attributes[i]);

		distance = Math.sqrt(distance);

		//if distance is less than neighbor threshold records are neighbors,
		//otherwise records are not neighbors
		if (distance <= delta)
			return 1;
		else
			return 0;
	}

	/************************************************************************/

	//Method initializes clusters of records
	private void initializeClusters()
	{
		//create array of cluster labels
		clusters = new int[numberRecords];

		//assign cluster -1 to all records
		for (int i = 0; i < numberRecords; i++)
			clusters[i] = -1;
	}

	/************************************************************************/

	//Method assigns cluster name to a record and all records connected to it
	//using breadth first traversal
	private void assignCluster(int index, int clusterName)
	{
		//assign cluster name to record
		clusters[index] = clusterName;

		//list used in traversal
		LinkedList<Integer> list = new LinkedList<Integer>();

		//put record into list
		list.addLast(index);

		//while list has records
		while (!list.isEmpty())
		{
			//remove first record from list
			int i = list.removeFirst();

			//find neighbors of record which have no cluster names
			for (int j = 0; j < numberRecords; j++)
				if (matrix[i][j] == 1 && clusters[j] == -1)
				{
					//assign cluster name to neighbor
					clusters[j] = clusterName;

					//add neighbor to list
					list.addLast(j);
				}
		}
	}

	/************************************************************************/

	//Method writes records and their clusters to output file
	public void display(String outputFile) throws IOException
	{
		PrintWriter outFile = new PrintWriter(new FileWriter(outputFile));

		//for each cluster
		for(int c = 0; c < numberClusters; c++)
		{
			//for each record
			for (int i = 0; i < numberRecords; i++)
			{
				if(clusters[i] == c)
				{
					//write attributes of record
					for (int j = 0; j < numberAttributes; j++)
						outFile.print(records.get(i).attributes[j] + " ");
		
					//write cluster label
					outFile.println(clusters[i]+1);
				}
			}
		}

		outFile.close();
	}

	/************************************************************************/
	
	//Method displays number of clusters
	public void numberClusters()
	{
		System.out.println("Number of clusters : " + numberClusters + " .");
	}
	
	/************************************************************************/

	//Method get centroids of clusters
	private double[][] getCentroids()
	{
		//centroids result
		double[][] centroids = new double[numberClusters][numberAttributes];
		
		//create array of cluster sums
		double[][] clusterSum = new double[numberClusters][numberAttributes];
		
		for (int i = 0; i < numberClusters; i++)
			for (int j = 0; j < numberAttributes; j++)
			{
				clusterSum[i][j] = 0;
				centroids[i][j] = 0;
			}

		//create array of cluster sizes and initialize
		int[] clusterSize = new int[numberClusters];
		for (int i = 0; i < numberClusters; i++)
			clusterSize[i] = 0;

		//for each record
		for (int i = 0; i < numberRecords; i++)
		{
			//find cluster of record
			int cluster = clusters[i];

			//add record to cluster sum
			clusterSum[cluster] = sum(clusterSum[cluster], records.get(i).attributes);

			//increment cluster size
			clusterSize[cluster] += 1;
		}

		//find centroid of each cluster
		for (int  i = 0; i < numberClusters; i++)
			centroids[i] = scale(clusterSum[i], 1.0/clusterSize[i]);
		
		return centroids;
	}

	/************************************************************************/

	//Method finds distance between two records
	private double distance(double[]u, double[] v)
	{
		double sum = 0;

		//find euclidean distance square between two records
		for (int i = 0; i < u.length; i++)
			sum += (u[i] - v[i])*(u[i] - v[i]);

		return sum;
	}

	/************************************************************************/

	//Method finds sum of two records
	private double[] sum(double[] u, double[] v)
	{
		double[] result = new double[u.length];

		//add corresponding attributes of records
		for (int i = 0; i < u.length; i++)
			result[i] = u[i] + v[i];

		return result;
	}

	/************************************************************************/

	//Method finds scaler multiple of a record
	private double[] scale(double[] u, double k)
	{
		double[] result = new double[u.length];

		//multiply attributes of record by scaler
		for (int i = 0; i < u.length; i++)
			result[i] = u[i]*k;

		return result;
	}

	/************************************************************************/
	
	//Method compute and display sum squared error
	public void displayError()
	{
		double sumError = 0;
		double[][] centroids = getCentroids();
		
		//go thru all records
		for(int i = 0; i < numberRecords; i++)
		{
			//find distance between record and its centroid
			double d = distance(records.get(i).attributes, centroids[clusters[i]]);
			
			sumError += d;
		}
		
		System.out.println("\nSum squared error : " + sumError);
	}
}