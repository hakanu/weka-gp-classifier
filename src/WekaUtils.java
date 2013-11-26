import wlsvm.WLSVM;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.ThresholdCurve;
import gp.GeneticProgramming;
import weka.classifiers.lazy.IBk;
import weka.classifiers.pmml.consumer.SupportVectorMachineModel;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaUtils {
	private static String fileName = "";
	private static String currentAlg = "";
	private static final String SEPERATOR = ",";
	private static String headers = (
			"fileName" + SEPERATOR +
			"currentAlg" + SEPERATOR +
			"class_index" + SEPERATOR +
			"numAttributes" + SEPERATOR +
			"correct" + SEPERATOR +
			"incorrect" + SEPERATOR +
			"pctCorrect" + SEPERATOR +
			"pctIncorrect" + SEPERATOR +
			"numPredictions" + SEPERATOR +
			"truePositiveRate" + SEPERATOR +
			"trueNegativeRate" + SEPERATOR +
			"precision" + SEPERATOR +
			"recall" + SEPERATOR +
			"numFalsePositives" + SEPERATOR +
			"numFalseNegatives" + SEPERATOR +
			"rocArea" + SEPERATOR +
			"classifierBuildElapsedTime" + SEPERATOR +
			"timeElapsed" + SEPERATOR +
			"numRunTimes" + SEPERATOR
			);
	
	private static String highLevelReportingHeaders = (
			"currentAlg" + SEPERATOR +
			"fileNameRefined" + SEPERATOR +
			"numAttributes" + SEPERATOR +
			"weightedAreaUnderROC" + SEPERATOR +
			"weightedFalseNegativeRate" + SEPERATOR +
			"weightedFalsePositiveRate" + SEPERATOR +
			"weightedFMeasure" + SEPERATOR +
			"weightedPrecision" + SEPERATOR +
			"weightedRecall" + SEPERATOR +
			"weightedTrueNegativeRate" + SEPERATOR +
			"weightedTruePositiveRate" + SEPERATOR +
			"classifierBuildElapsedTime" + SEPERATOR +
			"timeElapsed" + SEPERATOR +
			"numRunTimes" + SEPERATOR
			);
	public static double classifierBuildElapsedTime = 0.0;
	public static StringBuilder highLevelReporting = new StringBuilder();
	public static StringBuilder lowLevelReporting = new StringBuilder();
	
	public static Instances readArffFile(String fileName) {
		System.out.println("Read arff file... from : " + fileName);
		WekaUtils.fileName = fileName;
		try {
			DataSource source = new DataSource(fileName);
			Instances data = source.getDataSet();
			// Setting class attribute if the data format does not provide
			// this information.
			// For example, the XRFF format saves the class attribute
			// information as well.
			if (data.classIndex() == -1) {
				data.setClassIndex(data.numAttributes() - 1);
			}
			return data;
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("Finished reading arff file...\n\n");
		return null;
	}

	public static J48 buildJ48TreeClassifier(Instances data) {
		currentAlg = "J48";
		System.out.println("Running " + currentAlg + " tree classifier...");
		String[] options = new String[4];
		options[0] = "-C";
		options[1] = "0.25";
		options[2] = "-M";
		options[3] = "2";
		J48 tree = new J48();
		try {
			long classifierBuildStartTime = System.nanoTime();
			tree.setOptions(options);     // set the options
			tree.buildClassifier(data);
			long classifierBuildEndTime = System.nanoTime();
			double classifierBuildElapsedTime = (
					(classifierBuildEndTime - classifierBuildStartTime) /
					 1000000000.0);
			WekaUtils.classifierBuildElapsedTime = classifierBuildElapsedTime;
			System.out.println("+++++Classifier building ended in " +
							   classifierBuildElapsedTime + " seconds");
		} catch (Exception e) {
			e.printStackTrace();
		}   // build classifier
		System.out.println(currentAlg + " classifier finished.\n\n");
		return tree;
	}
	
	public static NaiveBayes buildNaiveBayesClassifier(Instances data) {
		currentAlg = "NaiveBayes";
		System.out.println("Running " + currentAlg + " tree classifier...");
		String[] options = new String[4];
		NaiveBayes tree = new NaiveBayes();
		try {
			long classifierBuildStartTime = System.nanoTime();
//			tree.setOptions(options);     // set the options
			tree.buildClassifier(data);
			long classifierBuildEndTime = System.nanoTime();
			double classifierBuildElapsedTime = (
					(classifierBuildEndTime - classifierBuildStartTime) /
					 1000000000.0);
			WekaUtils.classifierBuildElapsedTime = classifierBuildElapsedTime;
			System.out.println("+++++Classifier building ended in " +
							   classifierBuildElapsedTime + " seconds");
		} catch (Exception e) {
			e.printStackTrace();
		}   // build classifier
		System.out.println(currentAlg + " classifier finished.\n\n");
		return tree;
	}

	public static WLSVM buildSVMClassifier(Instances data) {
		currentAlg = "WLSVM";
		System.out.println("Running "+ currentAlg + " function classifier...");
		String dataFile = "";
		String[] options = {
				new String("-t"),
				dataFile,
				new String("-x"),		// 5 folds CV
				new String("5"),
				new String("-i"),		//
				//---------------
				new String("-S"),		// WLSVM options
				new String("0"),		// Classification problem
				new String("-K"),       // RBF kernel
				new String("2"),
				new String("-G"),       // gamma
				new String("1"),
				new String("-C"),       // C
				new String("7"),
				new String("-Z"),       // normalize input data
				new String("1"),
				new String("-M"),       // cache size in MB
				new String("100")				
				};
		WLSVM svmFunc = new WLSVM();
		try {
			long classifierBuildStartTime = System.nanoTime();
			//System.out.println(Evaluation.evaluateModel(lib,options));
//			svmFunc.setOptions(options);     // set the options
			svmFunc.buildClassifier(data);
			long classifierBuildEndTime = System.nanoTime();
			double classifierBuildElapsedTime = (
					(classifierBuildEndTime - classifierBuildStartTime) /
					 1000000000.0);
			WekaUtils.classifierBuildElapsedTime = classifierBuildElapsedTime;
			System.out.println("+++++Classifier building ended in " +
							   classifierBuildElapsedTime + " seconds");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return svmFunc;
	}
	
	public static GeneticProgramming buildGeneticProgrammingFunctionClassifier(
			Instances data) {
		currentAlg = "GeneticProgramming";
		System.out.println("Running "+ currentAlg + " function classifier...");
		String[] options = new String[4];
		GeneticProgramming gpFunc = new GeneticProgramming();
		System.out.println("Created gp classifier object.");
		try {
			long classifierBuildStartTime = System.nanoTime();
//			gpFunc.setOptions(options);     // set the options
			gpFunc.buildClassifier(data);
			long classifierBuildEndTime = System.nanoTime();
			double classifierBuildElapsedTime = (
					(classifierBuildEndTime - classifierBuildStartTime) /
					 1000000000.0);
			WekaUtils.classifierBuildElapsedTime = classifierBuildElapsedTime;
			System.out.println("+++++Classifier building ended in " +
							   classifierBuildElapsedTime + " seconds");
		} catch (Exception e) {
			e.printStackTrace();
		}   // build classifier
		System.out.println(currentAlg + " classifier finished.\n\n");
		return gpFunc;
	}
	
	public static RandomForest buildRandomForestClassifier(Instances data) {
		currentAlg = "RandomForest";
		System.out.println("Running " + currentAlg + " tree classifier...");
		String[] options = new String[6];
		options[0] = "-I";
		options[1] = "10";
		options[2] = "-K";
		options[3] = "0";
		options[4] = "-S";
		options[5] = "1";
		RandomForest tree = new RandomForest();
		try {
			long classifierBuildStartTime = System.nanoTime();
			tree.setOptions(options);     // set the options
			tree.buildClassifier(data);
			long classifierBuildEndTime = System.nanoTime();
			double classifierBuildElapsedTime = (
					(classifierBuildEndTime - classifierBuildStartTime) /
					 1000000000.0);
			WekaUtils.classifierBuildElapsedTime = classifierBuildElapsedTime;
			System.out.println("+++++Classifier building ended in " +
							   classifierBuildElapsedTime + " seconds");
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println(currentAlg + " tree classifier finished.\n\n");
		return tree;
	}
	
	public static Id3 buildId3TreeClassifier(Instances data) {
		currentAlg = "Id3";
		System.out.println("\n\nRunning " + currentAlg + " tree classifier...");
		String[] options = new String[6];
		options[0] = "-I";
		options[1] = "10";
		options[2] = "-K";
		options[3] = "0";
		options[4] = "-S";
		options[5] = "1";
		Id3 tree = new Id3();
		try {
			long classifierBuildStartTime = System.nanoTime();
//			tree.setOptions(options);     // set the options
			tree.buildClassifier(data);
			long classifierBuildEndTime = System.nanoTime();
			double classifierBuildElapsedTime = (
					(classifierBuildEndTime - classifierBuildStartTime) /
					 1000000000.0);
			WekaUtils.classifierBuildElapsedTime = classifierBuildElapsedTime;
			System.out.println("+++++Classifier building ended in " +
							   classifierBuildElapsedTime + " seconds");
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println(currentAlg + " tree classifier finished.\n\n");
		return tree;
	}
	
	public static IBk buildIBkClassifier(Instances data) {
		currentAlg = "IBk";
		System.out.println("Running IBk " + currentAlg + " lazy classifier...");
		String[] options = new String[1];
		IBk ibk = new IBk();
		try {
			long classifierBuildStartTime = System.nanoTime();
//			ibk.setOptions(options);     // set the options
			ibk.buildClassifier(data);
			long classifierBuildEndTime = System.nanoTime();
			double classifierBuildElapsedTime = (
					(classifierBuildEndTime - classifierBuildStartTime) /
					 1000000000.0);
			WekaUtils.classifierBuildElapsedTime = classifierBuildElapsedTime;
			System.out.println("+++++Classifier building ended in " +
							   classifierBuildElapsedTime + " seconds");
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println(currentAlg + " lazy classifier finished.\n\n");
		return ibk;
	}
	
	//////////////// Generic classifier.
	public static Classifier buildClassifier(Instances data,
											 String classifierName) {
		Classifier classifier = null;
		String[] options = new String[20];
		if (classifierName.equals("NaiveBayes")) {
			classifier = new NaiveBayes();
		} else if (classifierName.equals("Id3")) {
			classifier = new Id3();
			options[0] = "-I";
			options[1] = "10";
			options[2] = "-K";
			options[3] = "0";
			options[4] = "-S";
			options[5] = "1";
		} else if (classifierName.equals("RandomForest")) {
			classifier = new RandomForest();
			options[0] = "-I";
			options[1] = "10";
			options[2] = "-K";
			options[3] = "0";
			options[4] = "-S";
			options[5] = "1";
		} else if (classifierName.equals("IBk")) {
			classifier = new IBk();
			options[0] = "-K";
			options[1] = "5";
		} else if (classifierName.equals("SVM")) {
			classifier = new WLSVM();
		} else if (classifierName.equals("GeneticProgramming")) {
			classifier = new GeneticProgramming();
		} else if (classifierName.equals("J48")) {
			classifier = new J48();
			options[0] = "-C";
			options[1] = "0.25";
			options[2] = "-M";
			options[3] = "2";
		} else {
			System.out.println("Given classifier name " + classifierName +
							   " can not be found");
			return classifier;
		}
		currentAlg = classifierName;
		System.out.println("Building " + currentAlg + " classifier...");
		long classifierBuildStartTime = System.nanoTime();
//		classifier.setOptions(options);
		try {
			classifier.buildClassifier(data);
		} catch (Exception e) {
			e.printStackTrace();
		}
		long classifierBuildEndTime = System.nanoTime();
		double classifierBuildElapsedTime = (
				(classifierBuildEndTime - classifierBuildStartTime) /
				 1000000000.0);
		WekaUtils.classifierBuildElapsedTime = classifierBuildElapsedTime;
		System.out.println(currentAlg + " classifier building finished in " +
						   classifierBuildElapsedTime + " seconds");
		return classifier;
	}
	//////////////////
	
	public static String evaluateCrossValidate(
			Instances data, Classifier tree) {
		System.out.println("\n\nEvaluate with cross validation...");
		StringBuilder sb = new StringBuilder();
		Evaluation eval = null;
		try {
			eval = new Evaluation(data);
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("Evaluation object created");
		
		double timeElapsed = 0;
		try {
			long validatorBuildStartTime = System.nanoTime();
			eval.crossValidateModel(tree, data, 10, new Random(1));
			long validatorBuildEndTime = System.nanoTime();
			timeElapsed = (
					(validatorBuildEndTime - validatorBuildStartTime) /
					 1000000000.0);
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("Cross validate model finished in " + timeElapsed +
						   " seconds");

		int numAttributes = data.numAttributes();
		int numPredictions = eval.predictions().size();
		double correct = eval.correct();
		double incorrect = eval.incorrect();
		double pctCorrect = eval.pctCorrect();
		double pctIncorrect = eval.pctIncorrect();
		
		double weightedAreaUnderROC = eval.weightedAreaUnderROC();
		double weightedFalseNegativeRate = eval.weightedFalseNegativeRate();
		double weightedFalsePositiveRate = eval.weightedFalsePositiveRate();
		double weightedFMeasure = eval.weightedFMeasure();
		double weightedPrecision = eval.weightedPrecision();
		double weightedRecall = eval.weightedRecall();
		double weightedTrueNegativeRate = eval.weightedTrueNegativeRate();
		double weightedTruePositiveRate = eval.weightedTruePositiveRate();

		System.out.println("NumPredictions : " + numPredictions);
		System.out.println("Num attr : " + numAttributes);
		System.out.println("Num instances : " + data.numInstances());
		System.out.println("Num classes : " + data.numClasses());
		System.out.println("Correct   : " + correct);
		System.out.println("Incorrect : " + incorrect);
		System.out.println("Correct pct   : " + pctCorrect);
		System.out.println("Incorrect pct : " + pctIncorrect);
		
		System.out.println("weightedAreaUnderROC: " + weightedAreaUnderROC);
		System.out.println("weightedFMeasure: " + weightedFMeasure);
		System.out.println("weightedPrecision: " + weightedPrecision);
		System.out.println("weightedRecall: " + weightedRecall);
		System.out.println("weightedTruePositiveRate: " + weightedTruePositiveRate);
		System.out.println("weightedTrueNegativeRate: " + weightedTrueNegativeRate);
		System.out.println("weightedFalseNegativeRate: " + weightedFalseNegativeRate);
		System.out.println("weightedFalsePositiveRate: " + weightedFalsePositiveRate);

		String fileNameRefined = (
				fileName.replace("dataset/", "").replace("ext/", ""));

		for (int i = 0; i < data.numClasses(); i++) {
			System.out.println(i + ". class label");
			
			double truePositiveRate = eval.truePositiveRate(i);
			double trueNegativeRate = eval.trueNegativeRate(i);
			double precision = eval.precision(i);
			double recall = eval.recall(i);
			double numFalsePositives = eval.numFalsePositives(i);
			double numFalseNegatives = eval.numFalseNegatives(i);
			
			System.out.println("truePositiveRate : " + truePositiveRate);
			System.out.println("trueNegativeRate : " + trueNegativeRate);
			System.out.println("precision : " + precision);
			System.out.println("recall : " + recall);
			System.out.println("numFalsePositives : " + numFalsePositives);
			System.out.println("numFalseNegatives : " + numFalseNegatives);
			
			ThresholdCurve tc = new ThresholdCurve();
			Instances resultCurve = tc.getCurve(eval.predictions(), i);
			double rocArea = ThresholdCurve.getROCArea(resultCurve);
			System.out.println("ROCArea : " + rocArea);
			String localResults = (
					fileNameRefined + SEPERATOR +
					currentAlg + SEPERATOR +
					"class_" + i + SEPERATOR +
					numAttributes + SEPERATOR +
					correct + SEPERATOR +
					incorrect + SEPERATOR +
					pctCorrect + SEPERATOR +
					pctIncorrect + SEPERATOR +
					numPredictions + SEPERATOR +
					truePositiveRate + SEPERATOR +
					trueNegativeRate + SEPERATOR +
					precision + SEPERATOR +
					recall + SEPERATOR +
					numFalsePositives + SEPERATOR +
					numFalseNegatives + SEPERATOR +
					rocArea + SEPERATOR +
					WekaUtils.classifierBuildElapsedTime + SEPERATOR +  // DRY.
					timeElapsed + SEPERATOR);  // DRY.
			sb.append(localResults + "\n");
			System.out.println("********************\n");
		}
		
		highLevelReporting.append(
				currentAlg + SEPERATOR +
				fileNameRefined + SEPERATOR +
				numAttributes + SEPERATOR +
				weightedAreaUnderROC + SEPERATOR +
				weightedFalseNegativeRate + SEPERATOR +
				weightedFalsePositiveRate + SEPERATOR +
				weightedFMeasure + SEPERATOR +
				weightedPrecision + SEPERATOR +
				weightedRecall + SEPERATOR +
				weightedTrueNegativeRate + SEPERATOR +
				weightedTruePositiveRate + SEPERATOR +
				WekaUtils.classifierBuildElapsedTime + SEPERATOR +
				timeElapsed + SEPERATOR +
				"\n");
		
		System.out.println(headers);
		System.out.println(sb.toString());
		System.out.println("Evaluation of cross validation has finished...\n");
		return sb.toString();
	}
	
	public static String evaluateCrossValidateWithBestPick(
			Instances data, Classifier tree, int numRunTimes) {
		System.out.println("\n\nEvaluate with cross validation...");
		StringBuilder sb = new StringBuilder();
		Evaluation eval = null;
//		try {
//			eval = new Evaluation(data);
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
//		System.out.println("Evaluation object created");
		
		double timeElapsed = 0;
		Evaluation evalBest = null;
		try {
			long validatorBuildStartTime = System.nanoTime();
			for (int i = 0; i < numRunTimes; i++) {
				Evaluation ev = new Evaluation(data);
				ev.crossValidateModel(tree, data, 10, new Random(1));
				System.out.println("Recent validation result: " +
								   ev.weightedAreaUnderROC());
				if (evalBest == null) {
					evalBest = ev;
				} else {
					if (ev.weightedAreaUnderROC() >=
						evalBest.weightedAreaUnderROC()) {
						evalBest = ev;
					}
				}
				System.out.println("Best validation result: " +
						   		   evalBest.weightedAreaUnderROC());
			}
			long validatorBuildEndTime = System.nanoTime();
			timeElapsed = (
					(validatorBuildEndTime - validatorBuildStartTime) /
					 1000000000.0);
			eval = evalBest;
			System.out.println("Final Best validation result: " +
							   evalBest.weightedAreaUnderROC());
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("Cross validate model finished in " + timeElapsed +
						   " seconds with " + numRunTimes + " iterations.");

		int numAttributes = data.numAttributes();
		int numPredictions = eval.predictions().size();
		double correct = eval.correct();
		double incorrect = eval.incorrect();
		double pctCorrect = eval.pctCorrect();
		double pctIncorrect = eval.pctIncorrect();
		
		double weightedAreaUnderROC = eval.weightedAreaUnderROC();
		double weightedFalseNegativeRate = eval.weightedFalseNegativeRate();
		double weightedFalsePositiveRate = eval.weightedFalsePositiveRate();
		double weightedFMeasure = eval.weightedFMeasure();
		double weightedPrecision = eval.weightedPrecision();
		double weightedRecall = eval.weightedRecall();
		double weightedTrueNegativeRate = eval.weightedTrueNegativeRate();
		double weightedTruePositiveRate = eval.weightedTruePositiveRate();

		System.out.println("NumPredictions : " + numPredictions);
		System.out.println("Num attr : " + numAttributes);
		System.out.println("Num instances : " + data.numInstances());
		System.out.println("Num classes : " + data.numClasses());
		System.out.println("Correct   : " + correct);
		System.out.println("Incorrect : " + incorrect);
		System.out.println("Correct pct   : " + pctCorrect);
		System.out.println("Incorrect pct : " + pctIncorrect);
		
		System.out.println("weightedAreaUnderROC: " + weightedAreaUnderROC);
		System.out.println("weightedFMeasure: " + weightedFMeasure);
		System.out.println("weightedPrecision: " + weightedPrecision);
		System.out.println("weightedRecall: " + weightedRecall);
		System.out.println("weightedTruePositiveRate: " + weightedTruePositiveRate);
		System.out.println("weightedTrueNegativeRate: " + weightedTrueNegativeRate);
		System.out.println("weightedFalseNegativeRate: " + weightedFalseNegativeRate);
		System.out.println("weightedFalsePositiveRate: " + weightedFalsePositiveRate);

		String fileNameRefined = (
				fileName.replace("dataset/", "").replace("ext/", ""));

		for (int i = 0; i < data.numClasses(); i++) {
			System.out.println(i + ". class label");
			
			double truePositiveRate = eval.truePositiveRate(i);
			double trueNegativeRate = eval.trueNegativeRate(i);
			double precision = eval.precision(i);
			double recall = eval.recall(i);
			double numFalsePositives = eval.numFalsePositives(i);
			double numFalseNegatives = eval.numFalseNegatives(i);
			
			System.out.println("truePositiveRate : " + truePositiveRate);
			System.out.println("trueNegativeRate : " + trueNegativeRate);
			System.out.println("precision : " + precision);
			System.out.println("recall : " + recall);
			System.out.println("numFalsePositives : " + numFalsePositives);
			System.out.println("numFalseNegatives : " + numFalseNegatives);
			
			ThresholdCurve tc = new ThresholdCurve();
			Instances resultCurve = tc.getCurve(eval.predictions(), i);
			double rocArea = ThresholdCurve.getROCArea(resultCurve);
			System.out.println("ROCArea : " + rocArea);
			String localResults = (
					fileNameRefined + SEPERATOR +
					currentAlg + SEPERATOR +
					"class_" + i + SEPERATOR +
					numAttributes + SEPERATOR +
					correct + SEPERATOR +
					incorrect + SEPERATOR +
					pctCorrect + SEPERATOR +
					pctIncorrect + SEPERATOR +
					numPredictions + SEPERATOR +
					truePositiveRate + SEPERATOR +
					trueNegativeRate + SEPERATOR +
					precision + SEPERATOR +
					recall + SEPERATOR +
					numFalsePositives + SEPERATOR +
					numFalseNegatives + SEPERATOR +
					rocArea + SEPERATOR +
					WekaUtils.classifierBuildElapsedTime + SEPERATOR +  // DRY.
					timeElapsed + SEPERATOR +
					numRunTimes + SEPERATOR);  // DRY.
			sb.append(localResults + "\n");
			System.out.println("********************\n");
		}
		
		highLevelReporting.append(
				currentAlg + SEPERATOR +
				fileNameRefined + SEPERATOR +
				numAttributes + SEPERATOR +
				weightedAreaUnderROC + SEPERATOR +
				weightedFalseNegativeRate + SEPERATOR +
				weightedFalsePositiveRate + SEPERATOR +
				weightedFMeasure + SEPERATOR +
				weightedPrecision + SEPERATOR +
				weightedRecall + SEPERATOR +
				weightedTrueNegativeRate + SEPERATOR +
				weightedTruePositiveRate + SEPERATOR +
				WekaUtils.classifierBuildElapsedTime + SEPERATOR +
				timeElapsed + SEPERATOR +
				numRunTimes + SEPERATOR +
				"\n");
		
		System.out.println(headers);
		System.out.println(sb.toString());
		System.out.println("Evaluation of cross validation has finished...\n");
		return sb.toString();
	}
	
	public static String createSummary(StringBuilder sb, boolean isHighLevel) {
		System.out.println("Creating summary from string builder...");
		String summary = "";
		PrintWriter writer = null;
		try {
			writer = new PrintWriter("results_log.txt", "UTF-8");
			writer.print(sb.toString());
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		if (isHighLevel) {
			System.out.println(highLevelReportingHeaders);
		} else {
			System.out.println(headers);
		}
		System.out.println(sb.toString());
		System.out.println("Creating summary is done.");
		return summary;
	}
}
