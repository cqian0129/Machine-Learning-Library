import java.io.*;
import java.util.*;

public class NearestNeighbor {
    /**************************************************************************************************************/

    //Training record class
    private class Record
    {
        private double[] attributes;                    //attributes of record
        private int className;                          //class of record

        //Constructor of record
        private Record (double[] attributes, int className)
        {
            this.attributes = attributes;               //assign attributes
            this.className = className;                 //assign class
        }
        
    }

    /**************************************************************************************************************/

    private ArrayList<Record> records;                  //list of training records

    private int numberRecords;                          //number of training records
    private int numberAttributes;                       //number of attributes
    private int numberClasses;                          //number of classes

    private int numberNeighbors;                        //number of nearest neighbors
    private String majorityRule;                        //majority rule used

    /**************************************************************************************************************/

    //Constructor of classifier
    public NearestNeighbor ()
    {
        records = null;                                 //initialize records to empty

        numberRecords = 0;                              //number of records, attributes
        numberAttributes = 0;                           //classes are zero
        numberClasses = 0;

        numberNeighbors = 0;                            //nearest neighbors, majority rule
        majorityRule = null;                            //have no values
    }

    /**************************************************************************************************************/
    
    //return total number of records
    public int getNumberRecords(){
    	return this.numberRecords;
    	
    }
  
    /**************************************************************************************************************/

    //Method loads training records from training file
    public void loadTrainingData (String trainingFile) throws IOException
    {
        Scanner inFile = new Scanner(new File(trainingFile));

        //read number of records, attributes, classes
        numberRecords = inFile.nextInt();
        numberAttributes = inFile.nextInt();
        numberClasses = inFile.nextInt();

        //read neighbors, majority rule
        numberNeighbors = inFile.nextInt();
        majorityRule = inFile.next();

        //empty list of records
        records = new ArrayList<Record>();

        //for each record
        for (int i = 0; i < numberRecords; i++)
        {
            //create attributes array
            double[] attributeArray = new double[numberAttributes];

            //read attributes and convert them to numerical form
            for (int j = 0; j < numberAttributes; j++)
            {
                String label = inFile.next();
                attributeArray[j] = convert(label, j+1);
            }

            //read class and convert it to numerical form
            String label = inFile.next();
            int className = convert(label);

            //create record
            Record record = new Record(attributeArray, className);

            //add record to list of records
            records.add(record);
        }

        inFile.close();
    }

    /**************************************************************************************************************/

    //Method reads test records from test file and writes classes to classified file
    public void classifyData (String testFile, String classifiedFile) throws IOException
    {
        Scanner inFile = new Scanner(new File(testFile));
        PrintWriter outFile = new PrintWriter(new FileWriter(classifiedFile));

        //read number of records
        int numberRecords = inFile.nextInt();

        //for each record
        for (int i = 0; i < numberRecords; i++)
        {
            //create attribute array
            double[] attributeArray = new double[numberAttributes];

            //read attributes and convert them to numerical form
            for (int j = 0; j < numberAttributes; j++)
            {
                String label = inFile.next();
                attributeArray[j] = convert(label, j+1);
            }

            //find class of attribute
            int className = classify(attributeArray);

            //find class label and write to output file
            String label = convert(className);
            
            outFile.println(label);
        }

        inFile.close();
        outFile.close();
    }

    /**************************************************************************************************************/

    //Method finds class of given attributes
    private int classify(double[] attributes)
    {
        double[] distance = new double[numberRecords];
        int[] id = new int[numberRecords];

        //find distances between attributes and all records
        for (int i = 0; i < numberRecords; i++)
        {
            distance[i] = distance(attributes, records.get(i).attributes);
            id[i] = i;
        }

        //find the nearest neighbors
        nearestNeighbor(distance, id);

        //find majority class of neighbors
        int className = majority(id, attributes);

        //return class
        return className;
    }

    /**************************************************************************************************************/

    //Method finds the nearest neighbors
    private void nearestNeighbor(double[] distance, int[] id)
    {
    	int length = distance.length;
    	
        //sort the records by their distances and choose the closest neighbors
        for (int i = 0; i < numberNeighbors; i++)
            for (int j = i; j < length; j++)
                if (distance[i] > distance[j])
                {
                    double tempDistance = distance[i];
                    distance[i] = distance[j];
                    distance[j] = tempDistance;

                    int tempId = id[i];
                    id[i] = id[j];
                    id[j] = tempId;
                }
    }

    /**************************************************************************************************************/

    //Method finds the majority class of nearest neighbors
    private int majority (int[] id, double[] attributes)
    {
        double[] frequency = new double[numberClasses];

        //class frequencies are zero initially
        for (int i = 0; i < numberClasses; i++)
            frequency[i] = 0;

        //if unweighted majority rule is used
        if (majorityRule.equals("unweighted"))
        {
            //each neighbor contributes 1 to its class
            for (int i = 0; i < numberNeighbors; i++)
                frequency[records.get(id[i]).className - 1] += 1;
        }
        //if weighted majority rule is used
        else
        {
            //each neighbor contributes 1/distance to its class
            for (int i = 0; i < numberNeighbors; i++)
            {
                double d = distance(records.get(id[i]).attributes, attributes);
                frequency[records.get(id[i]).className - 1] += 1/(d + 0.001);
            }
        }

        //find majority class
        int maxIndex = 0;
        for (int i = 0; i < numberClasses; i++)
            if (frequency[i] > frequency[maxIndex])
                maxIndex = i;

        return maxIndex + 1;
    }

    /**************************************************************************************************************/

    //Method validates classifier using validation file and displays error rate
    public void validate (String validationFile) throws IOException
    {
        Scanner inFile = new Scanner(new File(validationFile));

        //read number of records
        int numberRecords = inFile.nextInt();

        //initially zero errors
        int numberErrors = 0;

        //for each record
        for (int i = 0; i < numberRecords; i++)
        {
            double[] attributeArray = new double[numberAttributes];

            //read attributes
            for (int j = 0; j < numberAttributes; j++)
            {
                String label = inFile.next();
                attributeArray[j] = convert(label, j+1);
            }

            //read actual class
            String label = inFile.next();
            int actualClass = convert(label);

            //find class predicted by classifier
            int predictedClass = classify(attributeArray);

            //error if predicted and actual classes do not match
            if (predictedClass != actualClass)
                numberErrors += 1;
        }

        //find and print error rate
        double errorRate = 100.0*numberErrors/numberRecords;
        System.out.println(errorRate + " percent error");

        inFile.close();
    }

    
    /**************************************************************************************************************/

    //Method that computes the training error of the classifier
    public int getTrainingError ()
    {
  
    	int numberErrors = 0;

        //for each training record
        for (int i = 0; i < numberRecords; i++)
        {
        	//get attributes
            double[] attributeArray = records.get(i).attributes;

            //read actual class
            int actualClass = records.get(i).className;

            //find class predicted by classifier
            int predictedClass = classify(attributeArray);

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
  
    	int numberErrors = 0;
    	
        //for each training record
        for (int i = 0; i < numberRecords; i++)
        {
        	boolean result = leaveOneOut(i);
        	
        	//if the predicted class != actual class
        	if(!result){
        		//System.out.println(i+1);
        		numberErrors ++;
        	}
        }
        
        return numberErrors;

    }
    
    /**************************************************************************************************************/

    //Method finds class of the record in given index without training it
    //if classified class = actual class, return true
    // else return false
    private boolean leaveOneOut(int index)
    {
    	double[] attributes = records.get(index).attributes;
    	int actualClass = records.get(index).className;
    	
    	//training records number = total records number - 1
    	int numberRecords = this.numberRecords - 1;
        double[] distance = new double[numberRecords];
        int[] id = new int[numberRecords];
        
        int j = 0; //keep track of index of distance and id

        //find distances between attributes and all records except the record in given index
        for (int i = 0; i < this.numberRecords; i++)
        {
        	if(i != index){
        		distance[j] = distance(attributes, records.get(i).attributes);
        		id[j] = i;
        		j++;
        	}  
        	//if i == index then skip
        }

        //find the nearest neighbors
        nearestNeighbor(distance, id);

        //find majority class of neighbors
        int predictedClass = majority(id, attributes);

        if(predictedClass == actualClass)
        	return true;
        else
        	return false;
    }
    
    
    /**************************************************************************************************************/

    //Method converts attribute values to numerical values. Hard coded for specific application
    private double convert (String label, int column)
    {
        double value;

        //convert sex attribute to 0/1
        if (column == 1)
        {
            if (label.equals("male"))
                value = 0;
            else
                value = 1;
        }
        //convert score attribute to [0 , 1] range
        else if (column == 2)
        {
            value = Double.valueOf(label);
            value = value/100;
        }
        //convert grade attribute to [0, 1] range
        else
        {
            if (label.equals("A"))
                value = 1.0;
            else if (label.equals("B"))
                value = 0.6;
            else
                value = 0.2;
        }

        return value;
    }

    /**************************************************************************************************************/

    //Method converts class labels to integer values. Hard coded for specific application
    private int convert (String label)
    {
        int value;

        if (label.equals("high"))
            value = 1;
        else if (label.equals("medium"))
            value = 2;
        else
            value = 3;

        return value;
    }

    /**************************************************************************************************************/

    //Method converts integer values to class labels. Hard coded for specific application
    private String convert(int value)
    {
        String label;
        if (value == 1)
            label = "high";
        else if (value == 2)
            label = "medium";
        else
            label = "low";

        return label;
    }

    /**************************************************************************************************************/

    //Method finds distance between two records. Hard coded for specific application
    private double distance(double[] u, double[] v)
    {
        double distance = 0;

        for (int i = 0; i < u.length; i++)
            distance = distance + (u[i] - v[i])*(u[i] - v[i]);

        distance = Math.sqrt(distance);

        return distance;
    }
}
