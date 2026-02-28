import { deleteHistory } from '../utils/api';

function HistoryTable({ history, onHistoryUpdate }) {
    const handleDelete = async (id) => {
        if (window.confirm('Are you sure you want to delete this prediction?')) {
            try {
                await deleteHistory(id);
                onHistoryUpdate();
            } catch (error) {
                console.error('Failed to delete:', error);
                alert('Failed to delete prediction');
            }
        }
    };

    const exportToCSV = () => {
        if (history.length === 0) {
            alert('No data to export');
            return;
        }

        const headers = ['ID', 'Emotion', 'Confidence', 'Filename', 'Timestamp'];
        const rows = history.map(item => [
            item.id,
            item.emotion,
            item.confidence,
            item.filename || 'Recording',
            item.timestamp
        ]);

        const csvContent = [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `emotion-history-${Date.now()}.csv`;
        a.click();
        window.URL.revokeObjectURL(url);
    };

    const emotionEmojis = {
        'Happy': '😊',
        'Sad': '😢',
        'Angry': '😠',
        'Fear': '😨',
        'Surprise': '😲',
        'Disgust': '🤢',
        'Neutral': '😐'
    };

    return (
        <div className="history-table">
            {history.length === 0 ? (
                <div className="empty-history">
                    <p className="empty-icon">📭</p>
                    <p className="empty-text">No predictions yet. Upload or record audio to get started!</p>
                </div>
            ) : (
                <>
                    <div className="table-header">
                        <h3>Total Predictions: {history.length}</h3>
                        <button className="export-button" onClick={exportToCSV}>
                            📥 Export CSV
                        </button>
                    </div>

                    <div className="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Emotion</th>
                                    <th>Confidence</th>
                                    <th>File</th>
                                    <th>Timestamp</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {history.map((item, index) => (
                                    <tr key={item.id}>
                                        <td>{index + 1}</td>
                                        <td className="emotion-cell">
                                            <span className="table-emoji">{emotionEmojis[item.emotion]}</span>
                                            <strong>{item.emotion}</strong>
                                        </td>
                                        <td>
                                            <span className="confidence-badge">
                                                {item.confidence.toFixed(1)}%
                                            </span>
                                        </td>
                                        <td className="filename-cell">
                                            {item.filename || '🎙️ Recording'}
                                        </td>
                                        <td className="timestamp-cell">
                                            {new Date(item.timestamp).toLocaleString()}
                                        </td>
                                        <td>
                                            <button
                                                className="delete-button"
                                                onClick={() => handleDelete(item.id)}
                                                title="Delete"
                                            >
                                                🗑️
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </>
            )}
        </div>
    );
}

export default HistoryTable;
