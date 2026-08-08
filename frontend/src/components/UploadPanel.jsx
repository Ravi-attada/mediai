import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'

export default function UploadPanel({ onFileSelect }) {
  const [fileName, setFileName] = useState('')

  const onDrop = useCallback(files => {
    if (files[0]) {
      setFileName(files[0].name)
      onFileSelect(files[0])
    }
  }, [onFileSelect])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.dcm'] },
    maxFiles: 1
  })

  return (
    <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
      <input {...getInputProps()} />
      <div className="dropzone-icon">🫁</div>
      {isDragActive
        ? <p>Drop your X-ray here...</p>
        : <p>Drag & drop a chest X-ray<br /><small>PNG, JPG, DICOM supported</small></p>
      }
      {fileName && <p className="file-name">✅ {fileName}</p>}
    </div>
  )
}
