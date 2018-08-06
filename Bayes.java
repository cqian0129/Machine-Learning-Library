import java.io.*;
import java.text.*;
import java.util.*;

//Bayes classifier class
public class Bayes {
	
	/************************************************************************/
	
	//Record class
	private class Record
	{
		private int[] attributes;			//attributes of record
		private int className;				//class of record
		private double confidence;			//classification confidence
		
		//Constructor of Record class
		private Record(int[] attributes, int className, double condidence)
		{
			this.attributes = attributes;	//assign attributes
			this.className = className;		//assign class
			this.confidence = condidence;	//assign confidence
		}
	}
	
	/************************************************************************/
	
	private int numberRecords;				//number of records
	private int numberAttributes;			//number of attributes
	private int numberClasses;				//number of classes
	
	private ArrayList<Record> records;		//list of records
	private int[] attributeValues;			//number of attribute value
	
	double[][][] table; 					//conditional probabilities
	double[] classTable;					//class probabilities
	
	/************************************************************************/

	//Constructor of Bayes class
	public Bayes()
	{
		numberRecords = 0;					//set number of records, attributes,
		numberAttributes = 0;				//and classes to zero
		numberClasses = 0;
		
		records = null;						//set records and attribute values
		attributeValues = null;				//to empty
		
		table = null;						//set probability tables to empty
		classTable = null;
	}
	
	/************************************************************************/

	//Method loads records from training file
	public void loadTrainingData(String trainingFile) throws IOException
	{
		Scanner inFile = new Scanner(new File(trainingFile));
		
		//read number of records, attributes, classes
		numberRecords = inFile.nextInt();
		numberAttributes = inFile.nextInt();
		numberClasses = inFile.nextInt();
		
		//read number of attribute values
		attributeValues = new int[numberAttributes];
		for (int i = 0; i < numberAttributes; i++)
			attributeValues[i] = inFile.nextInt();
	
		//list of records
		records = new ArrayList<Record>();
		
		//read each record
		for (int i = 0; i < numberRecords; i++)
		{
			//create attribute array
			int[] attributeArray = new int[numberAttributes];
			
			//read attributes and convert them to numerical form
			for (int j = 0; j < numberAttributes; j++)
			{
				String label = inFile.next();
				attributeArray[j] = convert(label, j+1);
			}
			
			//read class and convert it to numerical form
			String label = inFile.next();
			int className = convert(label);
			
			//create record, confidence is 1
			Record record = new Record(attributeArray, className, 1);
			
			//add record to list of records
			records.add(record);
		}
		
		inFile.close();
	}
	
	/************************************************************************/

	//Method computes probability values necessary for Bayes classification
	public void computeProbability()
	{
		//compute class probabilities
		computeClassTable();
		
		//compute conditional probabilities
		computeTable();
	}
	
	/************************************************************************/

	//Method computes class probabilities
	private void computeClassTable()
	{
		classTable = new double[numberClasses];
		
		//initialize class frequencies
		for (int i = 0; i < numberClasses; i++)
			classTable[i] = 0;
		
		//compute class frequencies
		for (int i = 0; i < numberRecords; i++)
			classTable[records.get(i).className-1] += 1;
		
		//normalize class frequencies
		for (int i = 0; i < numberClasses; i++)
			classTable[i] /= numberRecords;
	}
	
	/************************************************************************/

	//Method computes conditional probabilities
	private void computeTable()
	{
		//array to store conditional probabilities
		table = new double[numberAttributes][][];
		
		//compute conditional probabilities of each attribute
		for (int i = 0; i < numberAttributes; i++)
			compute(i+1);
	}
	
	/************************************************************************/

	//Method computes conditional probabilities of an attribute
	private void compute(int attribute)
	{
		//find number of attribute values
		int attributeValues = this.attributeValues[attribute-1];
		
		//create array to hold conditional probabilities
		table[attribute-1] = new double[numberClasses][attributeValues];
		
		//initialize conditional probabilities
		for (int i = 0; i < numberClasses; i++)
			for (int j = 0; j < attributeValues; j++)
				table[attribute-1][i][j] = 0;
		
		//compute class-attribute frequencies
		for (int k = 0; k < numberRecords; k++)
		{
			int i = records.get(k).className - 1;
			int j = records.get(k).attributes[attribute-1] - 1;
			table[attribute-1][i][j] += 1;
		}
		
		//compute conditional probabilities using laplace correction
		for (int i = 0; i < numberClasses; i++)
			for (int j = 0; j < attributeValues; j++)
			{
				double value = (table[attribute-1][i][j] + 1)/
								(classTable[i]*numberRecords + attributeValues);
				table[attribute-1][i][j] = value;
			}
	}
	
	/************************************************************************/

	//Method classifies an attribute, return a record with predicted class
	//and confidence
	private Record classify(int[] attributes)
	{
		double maxProbability = 0;
		double confidence = 0;
		int maxClass = 0;
		
		//for each class
		for (int i = 0; i < numberClasses; i++)
		{
			//find conditional probability of class given the attribute
			double probability = findProbability(i+1, attributes);
			
			//add to confidence
			confidence += probability;
			
			//choose the class with the maximum probability
			if (probability > maxProbability)
			{
				maxProbability = probability;
				maxClass = i;
			}
		}
		
		confidence = maxProbability / confidence;
		
		//construct result record
		Record result = new Record(attributes, maxClass + 1, confidence);
		
		return result;
	}
	
	/************************************************************************/

	//Method computes conditional probability of a class for given attributes
	private double findProbability(int className, int[] attributes)
	{
		double value;
		double product = 1;
		
		//find product of conditional probabilities stored in stable
		for (int i = 0; i < numberAttributes; i++)
		{
			value = table[i][className-1][attributes[i]-1];
			product = product*value;
		}
		
		//multiply product and class probability
		return product*classTable[className-1];
	}
	
	/************************************************************************/

	//Method reads test records from test file and writes classes
	//to classified file
	public void classifyData(String testFile, String classifiedFile)
	throws IOException
	{
		Scanner inFile = new Scanner(new File(testFile));
		PrintWriter outFile = new PrintWriter(new FileWriter(classifiedFile));
		
		//read number of records
		int numberRecords = inFile.nextInt();
		
		//read and classify each record
		for (int i = 0; i < numberRecords; i++)
		{
			//create attribute array
			int[] attributeArray = new int[numberAttributes];
			
			//read attributes and convert them to numerical form
			for (int j = 0; j < numberAttributes; j++)
			{
				String label = inFile.next();
				attributeArray[j] = convert(label, j+1);
			}
			
			//find class and confidence of attribute
			Record result = classify(attributeArray);
			int className = result.className;
			double confidence = result.confidence;
			
			//write class label and confidence to file
			String label = convert(className);
			outFile.println(label + "  " + confidence);
		}
		
		inFile.close();
		outFile.close();
	}
	

	/************************************************************************/
	
	//Method computes training error
	public void getTrainingError()
	{
		//record errors number
		int error = 0;
		
		//for each training record
        for (int i = 0; i < numberRecords; i++)
        {
        	//get attributes
            int[] attributeArray = records.get(i).attributes;

            //read actual class
            int actualClass = records.get(i).className;

            //find class predicted by classifier
            int predictedClass = classify(attributeArray).className;

            //error if predicted and actual classes do not match
            if (predictedClass != actualClass)
                error += 1;
        }
        
        System.out.println("Training error number: " + error);
        System.out.println("Training error rate: " + ((double)error/numberRecords));	
        System.out.println();
	}
	
	/************************************************************************/
	
	//Method computes validation error using leave one out method
	public void getValidationError()
	{	
		int error = 0;
    	
        //for each training record
        for (int i = 0; i < numberRecords; i++)
        {
        	boolean result = leaveOneOut();
        	
        	//if the predicted class != actual class
        	if(!result)
        		error ++;
        }
        
        System.out.println("Validation error number: " + error);
        System.out.println("Validation error rate: " + ((double)error/numberRecords));
        System.out.println();
        
        //restore tables
        computeProbability();
	}
	
	/************************************************************************/

    //Method finds class of the first record without training it
    //if classified class = actual class, return true, else return false
	//after testing, add it back to the end of the list
    private boolean leaveOneOut()
    {
    	//get first record
    	Record record = records.get(0); 
    	int[] attributes = record.attributes;
    	int actualClass = record.className;
    	
    	//remove it from the list
    	records.remove(0);
    	numberRecords --;
    	
    	//compute probability values
    	computeProbability();
    	
    	//classify the data
    	int predictedClass = classify(attributes).className;
    	
    	//add the record back to the end of the list
    	records.add(record);
    	numberRecords ++;
    	
        if(predictedClass == actualClass)
        	return true;
        else
        	return false;
    }
	
    /************************************************************************/
    
    //Method prints Laplace adjusted conditional probabilities tables
    public void printTable()
    {
    	//number format, keep only 3 digits
    	DecimalFormat df =new DecimalFormat("0.000");  
    	
    	//header
    	System.out.println();
    	System.out.println("Conditional Probabilities Tables");
    	System.out.println("---------------------------------------------");
    	
    	for(int i = 0; i < numberAttributes; i++)	//for each attribute
    	{
    		System.out.println("Attribute " + (i+1) + ":");
    		
    		//column contains class value, row contains attribute values
    		System.out.print("\t"); 	
    		
    		//print first row, label of attribute values
    		for(int k = 0; k < attributeValues[i]; k++)
    			System.out.print(convert(k+1, i+1) + "\t");
    		System.out.println();
    		
    		for(int j = 0; j < numberClasses; j++)
    		{
    			System.out.print(convert(j+1) + "\t");
    			for(int k = 0; k < attributeValues[i]; k++)
    				System.out.print(df.format(table[i][j][k]) + "\t");
    			System.out.println();
    		}
    		
    		System.out.println("---------------------------------------------");
    	}
    }
    
    /************************************************************************/
    
    //Method prints class probabilities table
    public void printClassTable()
    {
    	//number format, keep only 3 digits
    	DecimalFormat df =new DecimalFormat("0.000");  
    	
    	//header
    	System.out.println();
    	System.out.println("Class Probabilities Table");
    	System.out.println("---------------------------------------------");
    	
    	System.out.println("Class\tProbability");
		
		for(int i = 0; i < numberClasses; i++)
		{
			System.out.print(convert(i+1) + "\t");
			System.out.println(df.format(classTable[i]));
		}
		
		System.out.println("---------------------------------------------");
    }
    
	/************************************************************************/

	//Method converts attribute labels to numerical values
	private int convert(String label, int column)
	{
		int value;
		
		if (column == 1 || column == 4)
			value = Integer.valueOf(label) + 1;		//0, 1 + 1
		else 
			value = Integer.valueOf(label);

		//return numerical value
		return value;
	}
	
	/************************************************************************/

	//Method converts numerical values to attribute labels
	private String convert(int value, int column)
	{
		String label;
		
		if (column == 1 || column == 4)
			label =  String.valueOf(value - 1);		//0, 1
		else 
			label =  String.valueOf(value);;

		//return numerical value
		return label;
	}

	/************************************************************************/
	
	//Method converts class labels to numerical values
	private int convert(String label)
	{
		//return numerical value
		return Integer.valueOf(label);
	}
	
	/************************************************************************/

	//Method converts numerical values to class labels
	private String convert(int value)
	{
		//return class label
		return String.valueOf(value);	
	}
	
}
