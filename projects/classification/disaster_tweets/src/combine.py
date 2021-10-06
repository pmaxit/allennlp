import pandas as pd

def combine(test_file, output_file):
    test_file = pd.read_csv(test_file)
    output_file = pd.read_json(output_file,lines=True)

    submissions = pd.DataFrame({'id': test_file['id']})
    submissions['target'] = output_file.label.values

    return submissions
if __name__ == '__main__':
    df = combine("data/test/test.csv","data/output.csv")
    df.to_csv("submissions.csv",index=False)