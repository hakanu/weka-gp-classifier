/**
 * 
 */
import weka.core.Instances;

public class WekaOperations {
	private final String[] fileNames = {
//			"dataset/adult.data_normalized.arff",
//			"dataset/breast-cancer-wisconsin_normalized.arff",
//			"dataset/car.data.arff",
			"dataset/iris.arff",
//			"dataset/ext/contact-lenses.arff",
//			"dataset/ext/cpu.arff",
//			"dataset/ext/cpu.with.vendor.arff",
//			"dataset/ext/labor.arff",
//			"dataset/ext/segment-challenge.arff",
//			"dataset/ext/segment-test.arff",
//			"dataset/ext/soybean.arff",
//			"dataset/ext/weather.arff",
//			"dataset/ext/weather.nominal.arff"
			};
	
	private final String[] algorithmNames = {
			"J48",
			"RandomForest",
//			"Id3",
			"NaiveBayes",
			"IBk",
			"SVM",
			"GeneticProgramming",
	};
	
	public void classifyJ48Tree() {
		Instances data = WekaUtils.readArffFile(fileNames[0]);
		data.setClassIndex(data.numAttributes() - 1);
		StringBuilder outputLog = new StringBuilder();
		outputLog.append(WekaUtils.evaluateCrossValidate(data,
				WekaUtils.buildClassifier(data, "J48")));
		System.out.println(outputLog);
	}
	
	public void classifyNaiveBayes() {
		Instances data = WekaUtils.readArffFile(fileNames[0]);
		data.setClassIndex(data.numAttributes() - 1);
		StringBuilder outputLog = new StringBuilder();
		outputLog.append(WekaUtils.evaluateCrossValidate(data,
				WekaUtils.buildClassifier(data, "NaiveBayes")));
		System.out.println(outputLog);
	}
	
	public void classifyGeneticProgrammingFunction() {
		Instances data = WekaUtils.readArffFile(fileNames[0]);
		data.setClassIndex(data.numAttributes() - 1);
		StringBuilder outputLog = new StringBuilder();
		outputLog.append(WekaUtils.evaluateCrossValidate(data,
				WekaUtils.buildClassifier(data, "GeneticProgramming")));
		System.out.println(outputLog);
	}
	
	public void classifySVM() {
		Instances data = WekaUtils.readArffFile(fileNames[1]);
		data.setClassIndex(data.numAttributes() - 1);
		StringBuilder outputLog = new StringBuilder();
		outputLog.append(WekaUtils.evaluateCrossValidate(data,
				WekaUtils.buildClassifier(data, "SVM")));
		System.out.println(outputLog);
	}
	
	public void classifyIBkV2() {
		Instances data = WekaUtils.readArffFile(fileNames[2]);
		data.setClassIndex(data.numAttributes() - 1);
		StringBuilder outputLog = new StringBuilder();
		outputLog.append(WekaUtils.evaluateCrossValidate(data,
			WekaUtils.buildClassifier(data, "IBk")));
		System.out.println(outputLog);
	}
	
	public void classifyAllV2() {
		StringBuilder outputLog = new StringBuilder();
		// Iterate through the arff files.
		for (int i = 0; i < fileNames.length; i++) {
			Instances data = WekaUtils.readArffFile(fileNames[i]);
			data.setClassIndex(data.numAttributes() - 1);
			
			// Iterate through the algorithms.
			for (int j = 0; j < algorithmNames.length; j++) {
				outputLog.append(WekaUtils.evaluateCrossValidate(
						data, WekaUtils.buildClassifier(data,
														algorithmNames[j])));
				System.out.println("----------------------------------------");
			}
			System.out.println(outputLog.toString());
			System.out.println("++++++++++++++++++++++++++++++++++++++++++");
		}
		WekaUtils.createSummary(outputLog, false);
		System.out.println("\n\n/////////HIGH LEVEL////////////////\n\n");
		WekaUtils.createSummary(WekaUtils.highLevelReporting, true);
	}
	
	/**
	 * Be able to measure time better.
	 * For GP, choose the best performing one out of multiple cross validations.
	 */
	public void classifyAllV3WithMultipleRuns() {
		StringBuilder outputLog = new StringBuilder();
		// Iterate through the arff files.
		for (int i = 0; i < fileNames.length; i++) {
			Instances data = WekaUtils.readArffFile(fileNames[i]);
			data.setClassIndex(data.numAttributes() - 1);
			
			// Iterate through the algorithms.
			for (int j = 0; j < algorithmNames.length; j++) {
				outputLog.append(WekaUtils.evaluateCrossValidateWithBestPick(
						data, WekaUtils.buildClassifier(data,
														algorithmNames[j]),10));
				System.out.println("----------------------------------------");
			}
			System.out.println(outputLog.toString());
			System.out.println("++++++++++++++++++++++++++++++++++++++++++");
		}
		WekaUtils.createSummary(outputLog, false);
		System.out.println("\n\n/////////HIGH LEVEL////////////////\n\n");
		WekaUtils.createSummary(WekaUtils.highLevelReporting, true);
	}
}


