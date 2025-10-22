import { lazy } from 'react'

const ReactPlayer = lazy(() => 
  import('react-player/file').then(module => ({
    default: module.default || module
  }))
)

export default ReactPlayer
