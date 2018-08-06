import java.io.*;
import java.util.*;

//Decision tree class
public class DecisionTree {
    /*********************************************************************************************************/

    //Training record class
    private class Record
    {
        private int[] attributes;           	//attributes of record
        private int className;              	//class of record

        //constructor of record
        private Record(int[] attributes, int className)
        {
            this.attributes = attributes;       //assign attributes
            this.className = className;         //assign class
        }
    }

    /*********************************************************************************************************/

    //Decision tree node class
    private class Node
    {
        private String nodeType;            	//node type - internal or leaf
        private int condition;              	//condition if node is internal
        private int className;              	//class name if node is leaf
        
        //confidence
        private double coverage;				//number of records in the leaf / total records number
        private double accuracy;				//number of majority class records in the leaf / number records in the leaf
        
        private Node left;                  	//left branch
        private Node right;                 	//right branch

        //Constructor of node
        private Node(String nodeType, int value, Node left, Node right, double coverage, double accuracy)
        {
            this.nodeType = nodeType;               //assign node type
            this.left = left;                       //assign left branch
            this.right = right;                     //assign right branch
            
            this.coverage = coverage;				//assign coverage 
            this.accuracy = accuracy;				//assign accuracy 

            if (nodeType.equals("internal"))
            {                                       //if node is internal
                condition = value;                  //assign condition to node
                className = -1;                     //node has no class name
            }
            else
            {                                       //if node is leaf
                className = value;                  //assign class name to node
                condition = -1;                     //node has no condition
            }
        }
    }

    /*********************************************************************************************************/

    private Node root;                              //root of decision tree
    private ArrayList<Record> records;              //list of training records
    private ArrayList<Integer> attributes;          //list of attributes
    private int numberRecords;                      //number of training records
    private int numberAttributes;                   //number of attributes
    private int numberClasses;                      //number of classes

    /*********************************************************************************************************/

    //Constructor of decision tree
    public DecisionTree()
    {
        root = null;                                //initialize root, records,
        records = null;                             //attributes to empty
        attributes = null;
        numberRecords = 0;                          //number of records, attributes,
        numberAttributes = 0;                       //classes are zero
        numberClasses = 0;
    }
    /**************************************************************************************************************/
    
    //return total number of records
    public int getNumberRecords(){
    	return this.numberRecords;
    	
    }
    
    /*********************************************************************************************************/
    
    //Method builds tree for the whole training data
    public void buildTree(){
    	root = build(records, attributes);			//initial call to build method
    }
    
    /*********************************************************************************************************/

    //Method builds decision tree from given records and attributes
    //returns root of tree that is built
    private Node build(ArrayList<Record> records, ArrayList<Integer> attributes)
    {
        //root node is empty initially
        Node node = null;

        //if all records have same class
        if (sameClass(records))
        {
            //find class name
            int className = records.get(0).className;

            //records number in the leaf / total records number
            double coverage = getCoverage(records);
            
            //all records are the same class
            double accuracy =  1;
            
            //node is leaf with that class
            node = new Node("leaf", className, null, null,coverage,accuracy);
        }
        //if there are no attributes
        else if (attributes.isEmpty())
        {
        	
            //find majority class of records
            int className = majorityClass(records);

            //records number in the leaf / total records number
            double coverage = getCoverage(records);
            
            //records number of majority class / records number in the leaf
            double accuracy = getAccuracy(records, className);
            
            
            //node is leaf with that class
            node = new Node("leaf", className, null, null,coverage,accuracy);
        }
        else
        {
            //find best condition for current records and attributes
            int condition = bestCondition(records, attributes);

            //collect all records which have 0 for condition
            ArrayList<Record> leftRecords = collect(records, condition, 0);

            //collect all records which have 1 for condition
            ArrayList<Record> rightRecords = collect(records, condition, 1);

            //if either left records or right records is empty
            if (leftRecords.isEmpty() || rightRecords.isEmpty())
            {
            	
                //find majority class of records
                int className = majorityClass(records);
                
                //records number in the leaf / total records number
                double coverage = getCoverage(records);
                
                //records number of majority class / records number in the leaf
                double accuracy = getAccuracy(records, className);
                
                //node is leaf with that class
                node = new Node("leaf", className, null, null, coverage, accuracy);
            }
            else
            {
                //create copies of current attributes
                ArrayList<Integer> leftAttributes = copyAttributes(attributes);
                ArrayList<Integer> rightAttributes = copyAttributes(attributes);

                //remove best condition from current attributes
                leftAttributes.remove(new Integer(condition));
                rightAttributes.remove(new Integer(condition));

                //create internal node with best condition
                node = new Node("internal", condition, null, null, -1, -1);

                //create left subtree recursively
                node.left = build(leftRecords, leftAttributes);

                //create right subtree recursively
                node.right = build(rightRecords, rightAttributes);
            }
        }

        //return root node of tree that is built
        return node;
    }
    
    /*********************************************************************************************************/
    
    //get coverage, records number in the leaf / total records number
    private double getCoverage(ArrayList<Record> records){
    	return (double)records.size()/numberRecords;
    }
    
    /*********************************************************************************************************/
    
    //get accuracy, records number of majority class / records number in the leaf
    private double getAccuracy(ArrayList<Record> records,int majorityClass){
    	
    	double accuracy = 0;
        
        //count records number of majority class
        for (int i = 0; i < records.size(); i++)
        	if (records.get(i).className == majorityClass)
        		accuracy++;
        
        //records number of majority class / records number in the leaf
        return accuracy / records.size();
    	
    }
    
    /*********************************************************************************************************/

    //Method decides whether all records have the same class
    private boolean sameClass(ArrayList<Record> records)
    {
        //compare class of each record with class of first record
        for (int i = 0; i < records.size(); i++)
            if (records.get(i).className != records.get(0).className)
                return false;

        return true;
    }

    /*********************************************************************************************************/

    //Method finds the majority class of records
    private int majorityClass(ArrayList<Record> records)
    {
        int[] frequency = new int[numberClasses];       //frequency array

        for (int i = 0; i < numberClasses; i++)         //initialize frequencies
            frequency[i] = 0;

        for (int i = 0; i < records.size(); i++)        //find frequencies of classes
            frequency[records.get(i).className - 1] += 1;

        int maxIndex = 0;                               //find class with maximum
        for (int i = 0; i < numberClasses; i++)         //frequency
            if (frequency[i] > frequency[maxIndex])
                maxIndex = i;

        return maxIndex + 1;                            //return majority class
    }

    /*********************************************************************************************************/

    //Method collects records that have a given value for a given attribute
    private ArrayList<Record> collect(ArrayList<Record> records, int condition, int value)
    {
        //initialize collection
        ArrayList<Record> result = new ArrayList<Record>();

        //go thru records and collect those that have given value
        //for given attribute
        for (int i = 0; i < records.size(); i++)
            if (records.get(i).attributes[condition-1] == value)
                result.add(records.get(i));

        //return collection
        return result;
    }

    /*********************************************************************************************************/

    //Method makes copy of list of attributes
    private ArrayList<Integer> copyAttributes(ArrayList<Integer> attributes)
    {
        //initialize copy list
        ArrayList<Integer> result = new ArrayList<Integer>();

        //insert all attributes into copy list
        for (int i = 0; i < attributes.size(); i++)
            result.add(attributes.get(i));

        //return copy list
        return result;
    }

    /*********************************************************************************************************/

    //Method finds best condition for given records and attributes
    private int bestCondition(ArrayList<Record> records, ArrayList<Integer> attributes)
    {
        //evaluate first attribute
        double minValue = evaluate(records, attributes.get(0));
        int minIndex = 0;

        //go thru all attributes
        for (int i = 0; i < attributes.size(); i++)
        {
            double value = evaluate(records, attributes.get(i));        //evaluate attribute

            if (value < minValue)
            {                                                   //if value is less than
                minValue = value;                               //current minimum then
                minIndex = i;                                   //update minimum
            }
        }

        return attributes.get(minIndex);                        //return best attribute
    }

    /*********************************************************************************************************/

    //Method evaluates an attributes using weighted average entropy
    private double evaluate(ArrayList<Record> records, int attribute)
    {
        //collect records that have attribute value 0
        ArrayList<Record> leftRecords = collect(records, attribute, 0);

        //collect records that have attribute value 1
        ArrayList<Record> rightRecords = collect(records, attribute, 1);

        //find class entropy of left records
        double entropyLeft = entropy(leftRecords);

        //find class entropy of right records
        double entropyRight = entropy(rightRecords);

        //find weighted average entropy
        double average = entropyLeft*leftRecords.size()/records.size() +
                         entropyRight*rightRecords.size()/records.size();

        //return weighted average entropy
        return average;
    }

    /*********************************************************************************************************/

    //Method finds class entropy of records using gini measure
    private double entropy(ArrayList<Record> records)
    {
    	//get frequencies
    	
        double[] frequency = new double[numberClasses];         //frequency array

        for (int i = 0; i < numberClasses; i++)                 //initialize frequencies
            frequency[i] = 0;

        for (int i = 0; i < records.size(); i++)                //find class frequencies
            frequency[records.get(i).className - 1] += 1;

        double sum = 0;                                         //find sum of frequencies
        for (int i = 0; i < numberClasses; i++)
            sum = sum + frequency[i];

        for (int i = 0; i < numberClasses; i++)                 //normalize frequencies
            frequency[i] = frequency[i]/sum;

        double result = 1;
        
        //gini measure
        for (int i = 0; i < numberClasses; i++)              //1 - frequency * frequency
        	result = result - frequency[i]*frequency[i];
        
        return result;
    }
    
    //Method finds class entropy of records using class error measure
   /* private double entropy(ArrayList<Record> records)
    {
    	//get frequencies
    	
        double[] frequency = new double[numberClasses];         //frequency array

        for (int i = 0; i < numberClasses; i++)                 //initialize frequencies
            frequency[i] = 0;

        for (int i = 0; i < records.size(); i++)                //find class frequencies
            frequency[records.get(i).className - 1] += 1;

        double sum = 0;                                         //find sum of frequencies
        for (int i = 0; i < numberClasses; i++)
            sum = sum + frequency[i];

        for (int i = 0; i < numberClasses; i++)                 //normalize frequencies
            frequency[i] = frequency[i]/sum;

        double result = 0;
        
        //class measure
        double max = 0;
        for (int i = 0; i < frequency.length; i++)              //find maximum frequency
        	if(frequency[i] > max)
        		max = frequency[i];
        
        result = 1 - max;
 

        return result;
    }*/

    
    //Method finds class entropy of records using Shannon's entropy
    /*private double entropy(ArrayList<Record> records)
    {
    	//get frequencies
    	
        double[] frequency = new double[numberClasses];         //frequency array

        for (int i = 0; i < numberClasses; i++)                 //initialize frequencies
            frequency[i] = 0;

        for (int i = 0; i < records.size(); i++)                //find class frequencies
            frequency[records.get(i).className - 1] += 1;

        double sum = 0;                                         //find sum of frequencies
        for (int i = 0; i < numberClasses; i++)
            sum = sum + frequency[i];

        for (int i = 0; i < numberClasses; i++)                 //normalize frequencies
            frequency[i] = frequency[i]/sum;

        double result = 0;
        
        //shannon's entropy measure
        for (int i = 0; i < frequency.length; i++){      
        	
        	// result - frequency * log2frequency
        	result = result - frequency[i] * Math.log(frequency[i])/Math.log(2);
        	
        }

        return result;
    }*/


    /*********************************************************************************************************/

    //Method finds class of given attributes
    private Node classify(int[] attributes)
    {
        //start at root node
        Node current = root;

        //go down the tree
        while (current.nodeType.equals("internal"))
        {                                                   //if attribute value
            if (attributes[current.condition - 1] == 0)     //of condition is 0
                current = current.left;                     //go to left
            else
                current = current.right;                     //else go to right
        }
        
        return current;                            //return class name when reaching leaf
    }

    /*********************************************************************************************************/

    //Method loads training records from training file
    public void loadTrainingData(String trainingFile) throws IOException
    {
        Scanner inFile = new Scanner(new File(trainingFile));

        //read number of records, attributes, classes
        numberRecords = inFile.nextInt();
        numberAttributes = inFile.nextInt();
        numberClasses = inFile.nextInt();

        //empty list of records
        records = new ArrayList<Record>();

        //for each record
        for (int i = 0; i < numberRecords; i++)
        {
            //create attribute array
            int[] attributeArray = new int[numberAttributes];

            //for each attribute
            for (int j = 0; j < numberAttributes; j++)
            {
                //read attributes
                String label = inFile.next();
                //convert to binary
                attributeArray[j] = convert(label, j+1);
            }

            //read class and convert to integer value
            String label = inFile.next();
            int className = convert(label);

            //create record using attributes and class
            Record record = new Record(attributeArray, className);

            //add record to list
            records.add(record);
        }

        //create list of attributes
        attributes = new ArrayList<Integer>();
        for (int i = 0; i< numberAttributes; i++)
            attributes.add(i+1);

        inFile.close();
    }

    /*********************************************************************************************************/

    //Method reads test records from test file and writes classified records
    //to classified file
    public void classifyData(String testFile, String classifiedFile) throws IOException
    {
        Scanner inFile = new Scanner(new File(testFile));
        PrintWriter outFile = new PrintWriter(new FileWriter(classifiedFile));

        //read number of records
        int numberRecords = inFile.nextInt();

        //for each record
        for (int i = 0; i < numberRecords; i++)
        {
            //create attribute array
            int[] attributeArray = new int[numberAttributes];

            //read attributes and convert to binary
            for (int j = 0; j < numberAttributes; j++)
            {
                String label = inFile.next();
                attributeArray[j] = convert(label, j+1);
            }

            //find class of attributes
            Node current = classify(attributeArray);

            //find class name from integer value and write to output file
            String label = convert(current.className);
            outFile.print(label+"\t");
            outFile.print(current.coverage+"\t");
            outFile.print(current.accuracy+"\n");
        }

        inFile.close();
        outFile.close();
    }

    /*********************************************************************************************************/

    //Method validates decision tree using validation file and displays error rate
    public void validate(String validationFile) throws IOException
    {
        Scanner inFile = new Scanner(new File(validationFile));

        //read number of records
        int numberRecords = inFile.nextInt();

        //initialize number of errors
        int numberErrors = 0;

        //for each record
        for (int i = 0; i < numberRecords; i++)
        {
            //create attribute array
            int[] attributeArray = new int[numberAttributes];

            //read attributes and convert to binary
            for (int j = 0; j < numberAttributes; j++)
            {
                String label = inFile.next();
                attributeArray[j] = convert(label, j+1);
            }

            //read actual class from validation file
            String label = inFile.next();
            int actualClass = convert(label);

            //find class predicted by decision tree
            Node current = classify(attributeArray);
            int predictedClass = current.className;

            //error if predicted and actual classes do not match
            if (predictedClass != actualClass)
                numberErrors += 1;
        }

        //find and print error rate
        double errorRate = 100.0*numberErrors/numberRecords;
        System.out.println(errorRate + " percent error\n");

        inFile.close();
    }

    /*********************************************************************************************************/
    
    //Method that computes the training error of the classifier
    public int getTrainingError ()
    {
  
    	int numberErrors = 0;

        //for each training record
        for (int i = 0; i < numberRecords; i++)
        {
        	//get attributes
            int[] attributeArray = records.get(i).attributes;

            //read actual class
            int actualClass = records.get(i).className;

            //find class predicted by classifier
            Node current = classify(attributeArray);
            int predictedClass = current.className;

            //error if predicted and actual classes do not match
            if (predictedClass != actualClass){
            	
            	//System.out.println("Error class number: "+(i+1));
            	//System.out.println("Predicted class: " + convert(predictedClass));
            	//System.out.println("Actual class: " + convert(actualClass));
         
                numberErrors += 1;
            }
        }
        
        return numberErrors;

    }
    /**************************************************************************************************************/
    
    //Method that validates the classifier by applying the leave one out method on the training records
    //returns number of errors
    public int getLeaveOneOut ()
    {
    	//keep record of current root
    	Node currentRoot = root; 
    	
    	int numberErrors = 0;
    	
        //for each training record
        for (int i = 0; i < numberRecords; i++)
        {
        	boolean result = leaveOneOut(i);
        	if (!result){
        		//System.out.println(i+1);
        		numberErrors ++;
        	}
        }
        
        //restore the root
        root = currentRoot;
        
        return numberErrors;

    }
    
    /**************************************************************************************************************/
    
    //Method finds class of the first record without training it
    private boolean leaveOneOut(int index)
    {
    	//get the record that needs to leave out
    	Record tempRecord = records.get(index);
    	
    	//copy the records without the record in given index
    	ArrayList<Record> tempRecords = copyRecords(records, index);

    	//build decision tree with tempRecords
    	root = build(tempRecords, attributes);
    	
    	//find class predicted by classifier
        Node current = classify(tempRecord.attributes);
        int predictedClass = current.className;

        if (predictedClass == tempRecord.className)
        	return true;
        else
        	return false;
    }
    
    /*********************************************************************************************************/

    //Method makes copy of list of records without given index
    private ArrayList<Record> copyRecords(ArrayList<Record> records, int index)
    {
        //initialize copy list
        ArrayList<Record> result = new ArrayList<Record>();

        //insert records into copy list
        for (int i = 0; i < records.size(); i++)
            result.add(records.get(i));
        
        //remove record in given index
        result.remove(index);

        //return copy list
        return result;
    }
    
    /*********************************************************************************************************/
    
    //Method converts attribute labels to binary values, hard coded for specific application
    private int convert(String label, int column)
    {
        int value;

        //convert attribute labels to binary values
        if (column == 1)
            if (label.equals("highschool")) value = 0; else value = 1;
        else if (column == 2)
            if (label.equals("smoker")) value = 0; else value = 1;
        else if (column == 3)
            if (label.equals("married")) value = 0; else value = 1;
        else if (column == 4)
            if (label.equals("male")) value = 0; else value = 1;
        else
            if (label.equals("works")) value = 0; else value = 1;

        //return numerical value
        return value;
    }

    /*********************************************************************************************************/

    //Method converts class labels to integer values, hard coded for specific application
    private int convert(String label)
    {
        int value;

        //convert class labels to integer values
        if (label.equals("highrisk"))
            value = 1;
        else if (label.equals("mediumrisk"))
            value = 2;
        else if (label.equals("lowrisk"))
            value = 3;
        else
            value = 4;

        //return integer value
        return value;
    }

    /*********************************************************************************************************/

    //Method converts integer values to class labels, hard coded for
    //specific application
    private String convert(int value)
    {
        String label;

        //convert integer values to class labels
        if (value == 1)
            label = "highrisk";
        else if (value == 2)
            label = "mediumrisk";
        else if (value == 3)
            label = "lowrisk";
        else
            label = "undetermined";

        //return class label
        return label;
    }
}
