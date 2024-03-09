import React, { useState } from 'react';

const PredictorForm = () => {
    const [responses, setResponses] = useState(Array(10).fill('10'));

    const handleResponseChange = (index, value) => {
        const newResponses = [...responses];
        newResponses[index] = value;
        setResponses(newResponses);
    };

    const handleSubmit = () => {
        // Handle form submission
        console.log('Form submitted with responses:', responses);
    };

    return (
        <form onSubmit={handleSubmit}>
            {responses.map((value, index) => (
                <div key={index}>
                    <label htmlFor={`col${index}`}>Question {index + 1}:</label>
                    <select
                        name={`col${index}`}
                        value={value}
                        onChange={(e) => handleResponseChange(index, e.target.value)}
                    >
                        <option value="10">10</option>
                        <option value="20">20</option>
                        <option value="30">30</option>
                    </select><br />
                </div>
            ))}
            <button type="submit">Predict</button>
        </form>
    );
};

export default PredictorForm;
