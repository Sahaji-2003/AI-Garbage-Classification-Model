import React, { useState } from 'react';

const ImageUploadForm = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageChange = (e) => {
    setSelectedImage(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const formData = new FormData();
      formData.append('image', selectedImage);

      const response = await fetch('http://localhost:5000/classify-waste', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to classify waste');
      }

      const resultData = await response.json();
      setResult(resultData.result);
      setError(null);
    } catch (error) {
      setError(error.message);
      setResult(null);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleImageChange} />
        <button type="submit">Classify Waste</button>
      </form>

      {result && (
        <div>
          <h2>Result:</h2>
          <p>{result}</p>
        </div>
      )}

      {error && (
        <div>
          <h2>Error:</h2>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
};

export default ImageUploadForm;








  {/* <h3>Clasfication and Percentage of waste</h3>
                  <div className="skills-content">
                    <div className="progress">
                      <span className="skill">
                        Dry <i className="val">35%</i>
                      </span>
                      <div className="progress-bar-wrap">
                        <div
                          className="progress-bar"
                          role="progressbar"
                          aria-valuenow={100}
                          aria-valuemin={0}
                          aria-valuemax={100}
                        />
                      </div>
                    </div>
                    <div className="progress">
                      <span className="skill">
                        Organic <i className="val">25%</i>
                      </span>
                      <div className="progress-bar-wrap">
                        <div
                          className="progress-bar"
                          role="progressbar"
                          aria-valuenow={90}
                          aria-valuemin={0}
                          aria-valuemax={100}
                        />
                      </div>
                    </div>
                    <div className="progress">
                      <span className="skill">
                        {" "}
                        E-waste<i className="val">25%</i>
                      </span>
                      <div className="progress-bar-wrap">
                        <div
                          className="progress-bar"
                          role="progressbar"
                          aria-valuenow={75}
                          aria-valuemin={0}
                          aria-valuemax={100}
                        />
                      </div>
                    </div>
                    <div className="progress">
                      <span className="skill">
                        Biomedical<i className="val">15%</i>
                      </span>
                      <div className="progress-bar-wrap">
                        <div
                          className="progress-bar"
                          role="progressbar"
                          aria-valuenow={55}
                          aria-valuemin={0}
                          aria-valuemax={100}
                        />
                      </div>
                    </div>
                  </div> */}









                  // <h3>Clasfication and Percentage of waste</h3>
{/* {individualProbabilities.length > 0 && (
        <div>
         
          {individualProbabilities.map(([className, probability]) => (
            <p key={className}>{`${className}: ${probability.toFixed(2)}%`}</p>
          ))}
        </div>
      )} */}


{/* <div className="skills-content">
  {individualProbabilities.map(([className, probability]) => (
    <div className="progress" key={className}>
      <span className="skill">
        {className} <i className="val">{probability.toFixed(2)}%</i>
      </span>
      <div className="progress-bar-wrap">
        <div
          className="progress-bar"
          role="progressbar"
          style={{ width: `${probability}%` }}
          aria-valuenow={probability}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  ))} */}